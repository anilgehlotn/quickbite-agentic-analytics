import type { Config } from "tailwindcss";

/**
 * Tailwind configuration.
 *
 * Colours are declared as CSS variables in app/globals.css and referenced here
 * so the palette has a single source of truth. The theme is a light, warm,
 * paper-like canvas with one deep pine accent.
 *
 * Two deliberate overrides beyond the palette:
 *
 *   * `borderRadius` is capped at 6px across the whole scale, including the
 *     larger keys. Heavy rounding is the single strongest tell of a templated
 *     interface, and capping it here means no component can reintroduce it by
 *     reaching for `rounded-xl`. `rounded-full` is left alone for genuine
 *     pills — status badges and dots.
 *   * `boxShadow` carries exactly one entry. Structure is done with hairline
 *     borders; the question input is the only element that should float.
 */
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "var(--color-canvas)",
        surface: "var(--color-surface)",
        raised: "var(--color-raised)",
        sunken: "var(--color-sunken)",
        border: "var(--color-border)",
        "border-strong": "var(--color-border-strong)",
        ink: "var(--color-ink)",
        muted: "var(--color-muted)",
        faint: "var(--color-faint)",
        accent: "var(--color-accent)",
        "accent-hover": "var(--color-accent-hover)",
        "accent-soft": "var(--color-accent-soft)",
        "accent-line": "var(--color-accent-line)",
        "accent-fg": "var(--color-accent-fg)",
        positive: "var(--color-positive)",
        "positive-soft": "var(--color-positive-soft)",
        "positive-line": "var(--color-positive-line)",
        caution: "var(--color-caution)",
        "caution-soft": "var(--color-caution-soft)",
        "caution-line": "var(--color-caution-line)",
        negative: "var(--color-negative)",
        "negative-soft": "var(--color-negative-soft)",
        "negative-line": "var(--color-negative-line)",
      },
      borderRadius: {
        none: "0px",
        sm: "3px",
        DEFAULT: "4px",
        md: "5px",
        lg: "6px",
        xl: "6px",
        "2xl": "6px",
        "3xl": "6px",
        full: "9999px",
      },
      boxShadow: {
        input: "0 1px 2px rgba(26, 25, 23, 0.05)",
        pop: "0 2px 10px rgba(26, 25, 23, 0.10)",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "SF Mono",
          "Menlo",
          "monospace",
        ],
      },
      fontSize: {
        // A wide scale with real gaps between levels: a headline that dominates,
        // a lede that is clearly subordinate, and a label small enough that
        // uppercase and letter-spacing read as structure rather than shouting.
        display: ["2.75rem", { lineHeight: "1.08", letterSpacing: "-0.028em" }],
        lede: ["1.0625rem", { lineHeight: "1.65" }],
        label: ["0.6875rem", { lineHeight: "1.4" }],
        micro: ["0.625rem", { lineHeight: "1.5" }],
      },
      maxWidth: {
        content: "68rem",
      },
    },
  },
  plugins: [],
};

export default config;
