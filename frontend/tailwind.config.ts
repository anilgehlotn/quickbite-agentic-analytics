import type { Config } from "tailwindcss";

/**
 * Tailwind configuration.
 *
 * Colours are declared as CSS variables in app/globals.css and referenced here
 * so the palette has a single source of truth. The palette is a warm dark
 * neutral base with one amber accent, chosen to read as a considered product
 * rather than default Tailwind blue-on-gray.
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
        base: "var(--color-base)",
        surface: "var(--color-surface)",
        raised: "var(--color-raised)",
        border: "var(--color-border)",
        ink: "var(--color-ink)",
        muted: "var(--color-muted)",
        faint: "var(--color-faint)",
        accent: "var(--color-accent)",
        "accent-soft": "var(--color-accent-soft)",
        positive: "var(--color-positive)",
        caution: "var(--color-caution)",
        negative: "var(--color-negative)",
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
        display: ["3.25rem", { lineHeight: "1.05", letterSpacing: "-0.03em" }],
        lede: ["1.0625rem", { lineHeight: "1.6" }],
      },
      maxWidth: {
        content: "68rem",
      },
    },
  },
  plugins: [],
};

export default config;
