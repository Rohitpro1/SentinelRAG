/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#3525cd',
          container: '#4f46e5',
          dark: '#0f0069',
          light: '#e2dfff',
        },
        surface: {
          DEFAULT: '#f8f9ff',
          dim: '#cbdbf5',
          low: '#eff4ff',
          high: '#dce9ff',
          highest: '#d3e4fe',
          container: '#e5eeff',
        },
        onSurface: {
          DEFAULT: '#0b1c30',
          variant: '#464555',
        },
      },
      fontFamily: {
        sans: ['Geist', 'Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};
