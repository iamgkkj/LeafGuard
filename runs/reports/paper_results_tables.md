# LeafGuard AI — Experimental Results

| Metric              |   Baseline MobileNetV3-Small |   MobileNetV3-Small + CBAM |
|:--------------------|-----------------------------:|---------------------------:|
| Accuracy (%)        |                        99.77 |                      99.72 |
| Macro-F1 (%)        |                        99.77 |                      99.71 |
| Macro-Precision (%) |                        99.77 |                      99.71 |
| Macro-Recall (%)    |                        99.78 |                      99.71 |
| Parameters          |                   1556806.00 |                 1562524.00 |
| FLOPs (M)           |                        61.49 |                      62.22 |
| Model Size (MB)     |                         6.41 |                       6.49 |
| CPU FPS             |                       233.16 |                     247.65 |
| CPU Latency (ms)    |                         4.29 |                       4.04 |

## CBAM Gain Analysis

| Metric     |      Baseline |          CBAM |   Change_% |
|:-----------|--------------:|--------------:|-----------:|
| Accuracy   |        0.9977 |        0.9972 |    -0.0570 |
| Macro-F1   |        0.9977 |        0.9971 |    -0.0585 |
| Parameters |  1556806.0000 |  1562524.0000 |     0.3673 |
| FLOPs      | 61492856.0000 | 62218398.0000 |     1.1799 |
| Model Size |        6.4138 |        6.4858 |     1.1236 |
| CPU FPS    |      233.1595 |      247.6527 |     6.2160 |
