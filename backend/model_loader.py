"""
Model loading and management for LeafGuard AI.
Loads both baseline and CBAM models at startup.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict
import logging

from models import build_baseline, build_cbam
from preprocessing import IMAGE_SIZE


logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and inference for both models."""
    
    def __init__(self, checkpoint_dir: Path, device: Optional[str] = None):
        """
        Initialize model manager.

        Args:
            checkpoint_dir: Directory containing model checkpoints
            device: Device to load models on ('cuda', 'cpu', or None for auto)
        """
        # Resolve checkpoint directory to absolute path
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.baseline_model: Optional[nn.Module] = None
        self.cbam_model: Optional[nn.Module] = None
        self.class_names: Optional[list] = None

        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def load_class_names(self) -> list:
        """
        Load class names from the training split CSV.
        
        Returns:
            List of class names in label order
        """
        import pandas as pd
        
        # Splits directory is sibling of models directory
        splits_dir = self.checkpoint_dir.parent / "splits"
        train_csv = splits_dir / "train.csv"
        
        if not train_csv.exists():
            raise FileNotFoundError(f"Training split not found at {train_csv}")
        
        df = pd.read_csv(train_csv)
        self.class_names = sorted(df['class'].unique())
        logger.info(f"Loaded {len(self.class_names)} class names")
        
        return self.class_names
    
    def load_models(self) -> Dict[str, bool]:
        """
        Load both baseline and CBAM models.

        Returns:
            Dictionary with loading status for each model
        """
        status = {}

        # Load class names first
        if self.class_names is None:
            self.load_class_names()

        num_classes = len(self.class_names)

        # Load baseline model
        baseline_ckpt = self.checkpoint_dir / "baseline_best.pth"
        if baseline_ckpt.exists():
            try:
                self.baseline_model = build_baseline(num_classes)
                state_dict = torch.load(baseline_ckpt, map_location=self.device)
                # Filter out non-model keys (e.g., from thop FLOPs counting)
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('total_')}
                # Load with strict=False to handle potential key mismatches
                self.baseline_model.load_state_dict(state_dict, strict=False)
                self.baseline_model.to(self.device)
                self.baseline_model.eval()
                status['baseline'] = True
                logger.info(f"Baseline model loaded from {baseline_ckpt}")
            except Exception as e:
                logger.error(f"Failed to load baseline model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                status['baseline'] = False
        else:
            logger.warning(f"Baseline checkpoint not found at {baseline_ckpt}")
            status['baseline'] = False
        
        # Load CBAM model
        cbam_ckpt = self.checkpoint_dir / "cbam_best.pth"
        if cbam_ckpt.exists():
            try:
                self.cbam_model = build_cbam(num_classes)
                state_dict = torch.load(cbam_ckpt, map_location=self.device)
                # Filter out non-model keys (e.g., from thop FLOPs counting)
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('total_')}
                # Load with strict=False to handle potential key mismatches
                self.cbam_model.load_state_dict(state_dict, strict=False)
                self.cbam_model.to(self.device)
                self.cbam_model.eval()
                status['cbam'] = True
                logger.info(f"CBAM model loaded from {cbam_ckpt}")
            except Exception as e:
                logger.error(f"Failed to load CBAM model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                status['cbam'] = False
        else:
            logger.warning(f"CBAM checkpoint not found at {cbam_ckpt}")
            status['cbam'] = False
        
        return status
    
    def get_model(self, model_name: str) -> nn.Module:
        """
        Get a model by name.
        
        Args:
            model_name: 'baseline' or 'cbam'
            
        Returns:
            The requested model
            
        Raises:
            ValueError: If model name is invalid or model not loaded
        """
        if model_name == 'baseline':
            if self.baseline_model is None:
                raise ValueError("Baseline model not loaded")
            return self.baseline_model
        elif model_name == 'cbam':
            if self.cbam_model is None:
                raise ValueError("CBAM model not loaded")
            return self.cbam_model
        else:
            raise ValueError(f"Invalid model name: {model_name}")
    
    def get_classes(self) -> list:
        """Get the list of class names."""
        if self.class_names is None:
            self.load_class_names()
        return self.class_names
