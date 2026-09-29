/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        leaf: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
          950: '#052e16',
        },
        earth: {
          50: '#fdf8f6',
          100: '#f2e8e5',
          200: '#eaddd7',
          300: '#d6bfb6',
          500: '#a67c52',
          600: '#8c6240',
          700: '#7c5a3a',
          800: '#5a3f2a',
          900: '#4a3421'
        },
        sage: {
          50: '#f6f7f6',
          100: '#e3e8e3',
          200: '#c5d1c5',
          500: '#7a9a7a',
          700: '#4a6b4a',
        },
        moss: {
          50: '#f2f7f2',
          500: '#5b8c5a',
          700: '#3d6b3d',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Plus Jakarta Sans', 'Inter', 'sans-serif'],
      },
      boxShadow: {
        'soft': '0 2px 16px rgba(16, 87, 42, 0.06)',
        'soft-lg': '0 8px 32px rgba(16, 87, 42, 0.10)',
        'glass': '0 8px 32px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.6)',
        'inner-leaf': 'inset 0 1px 2px rgba(22,101,52,0.08)',
      },
      backdropBlur: {
        xs: '2px',
      },
      borderRadius: {
        '4xl': '2rem',
        '5xl': '2.5rem',
      },
      keyframes: {
        'fade-in': { '0%': { opacity: '0' }, '100%': { opacity: '1' } },
        'slide-up': { '0%': { opacity: '0', transform: 'translateY(8px)' }, '100%': { opacity: '1', transform: 'translateY(0)' } },
        'shimmer': { '0%': { backgroundPosition: '-200% 0' }, '100%': { backgroundPosition: '200% 0' } },
        'float': { '0%,100%': { transform: 'translateY(0)' }, '50%': { transform: 'translateY(-4px)' } },
        'pulse-slow': { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.7' } },
      },
      animation: {
        'fade-in': 'fade-in 0.4s ease-out',
        'slide-up': 'slide-up 0.4s ease-out',
        'shimmer': 'shimmer 2s linear infinite',
        'float': 'float 4s ease-in-out infinite',
        'pulse-slow': 'pulse-slow 3s ease-in-out infinite',
      }
    }
  },
  plugins: []
}
