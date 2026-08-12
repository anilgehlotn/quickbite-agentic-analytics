/**
 * Presentation helpers for figures and column names.
 *
 * Money is Indian-format throughout: the dataset is INR and a reader expecting
 * lakh/crore grouping reads 12,96,383 far faster than 1,296,383. Column labels
 * are humanised from the SQL aliases the agent chose, so the table headers read
 * as English without the backend having to send a second set of names.
 */

import type { CellValue, ResultRow } from "@/lib/types";

/** Column-name fragments that mark a value as money. */
const MONEY_TOKENS = [
  "revenue",
  "inr",
  "aov",
  "sales",
  "margin",
  "discount",
  "value",
  "cogs",
  "baseline",
  "delta_abs",
  "change_abs",
];

/** Column-name fragments that mark a value as a percentage. */
const PERCENT_TOKENS = ["pct", "percent", "share", "rate"];

/** Column-name fragments that mark a value as a plain count. */
const COUNT_TOKENS = ["orders", "count", "units", "qty", "quantity", "days"];

/** Words kept lowercase when humanising a column name. */
const MINOR_WORDS = new Set(["per", "of", "by", "vs", "and", "to"]);

/** Words that should be shown in full uppercase. */
const ACRONYMS = new Set(["inr", "aov", "sku", "id", "sql", "pct", "qty"]);

/**
 * Whether a column holds a monetary amount.
 *
 * @param column - The column name.
 * @returns True when the value should be rendered as INR.
 */
export function isMoneyColumn(column: string): boolean {
  const lowered = column.toLowerCase();
  if (PERCENT_TOKENS.some((token) => lowered.includes(token))) {
    return false;
  }
  return MONEY_TOKENS.some((token) => lowered.includes(token));
}

/**
 * Whether a column holds a percentage.
 *
 * @param column - The column name.
 * @returns True when the value should carry a percent sign.
 */
export function isPercentColumn(column: string): boolean {
  const lowered = column.toLowerCase();
  return PERCENT_TOKENS.some((token) => lowered.includes(token));
}

/**
 * Whether a column holds a countable quantity.
 *
 * @param column - The column name.
 * @returns True when the value is a plain integer count.
 */
export function isCountColumn(column: string): boolean {
  const lowered = column.toLowerCase();
  if (isMoneyColumn(column) || isPercentColumn(column)) {
    return false;
  }
  return COUNT_TOKENS.some((token) => lowered.includes(token));
}

/**
 * Format an INR amount in full, with Indian digit grouping.
 *
 * @param value - The amount in rupees.
 * @param decimals - Fraction digits to show.
 * @returns The formatted amount, prefixed with the rupee sign.
 */
export function formatInr(value: number, decimals = 2): string {
  return `₹${value.toLocaleString("en-IN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

/**
 * Format an INR amount abbreviated to the nearest sensible unit.
 *
 * Used on chart axes and metric cards, where ₹32.0L is legible and
 * ₹31,97,076.50 is not. Below a lakh the full figure is short enough to keep.
 *
 * @param value - The amount in rupees.
 * @returns The abbreviated amount, e.g. "₹32.0L" or "₹1.3Cr".
 */
export function formatInrCompact(value: number): string {
  const sign = value < 0 ? "-" : "";
  const magnitude = Math.abs(value);

  if (magnitude >= 10_000_000) {
    return `${sign}₹${(magnitude / 10_000_000).toFixed(1)}Cr`;
  }
  if (magnitude >= 100_000) {
    return `${sign}₹${(magnitude / 100_000).toFixed(1)}L`;
  }
  if (magnitude >= 1_000) {
    return `${sign}₹${(magnitude / 1_000).toFixed(1)}K`;
  }
  return `${sign}₹${magnitude.toLocaleString("en-IN", {
    maximumFractionDigits: 0,
  })}`;
}

/**
 * Format a count with Indian digit grouping.
 *
 * @param value - The count.
 * @returns The formatted integer.
 */
export function formatCount(value: number): string {
  return value.toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

/**
 * Format one cell for display, choosing the unit from its column name.
 *
 * @param column - The column the value came from.
 * @param value - The cell value.
 * @returns A display string; an em dash for null.
 */
export function formatCell(column: string, value: CellValue): string {
  if (value === null) {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (typeof value === "number") {
    if (isMoneyColumn(column)) {
      return formatInr(value, Number.isInteger(value) ? 0 : 2);
    }
    if (isPercentColumn(column)) {
      return `${value.toFixed(2)}%`;
    }
    if (isCountColumn(column) || Number.isInteger(value)) {
      return formatCount(value);
    }
    return value.toLocaleString("en-IN", { maximumFractionDigits: 2 });
  }
  return value;
}

/**
 * Turn a SQL column alias into a readable label.
 *
 * @param column - The column name, e.g. "revenue_inr" or "aov_2026_05".
 * @returns A humanised label, e.g. "Revenue INR".
 */
export function humaniseColumn(column: string): string {
  return column
    .split("_")
    .filter((part) => part.length > 0)
    .map((part, index) => {
      const lowered = part.toLowerCase();
      if (ACRONYMS.has(lowered)) {
        return lowered.toUpperCase();
      }
      if (index > 0 && MINOR_WORDS.has(lowered)) {
        return lowered;
      }
      return lowered.charAt(0).toUpperCase() + lowered.slice(1);
    })
    .join(" ");
}

/**
 * Format a duration for the trace panel.
 *
 * @param milliseconds - The duration.
 * @returns Milliseconds below a second, otherwise seconds to one decimal.
 */
export function formatDuration(milliseconds: number): string {
  if (milliseconds < 1_000) {
    return `${Math.round(milliseconds)}ms`;
  }
  return `${(milliseconds / 1_000).toFixed(1)}s`;
}

/**
 * Format a token count compactly.
 *
 * @param tokens - The number of tokens.
 * @returns e.g. "19.5k" or "840".
 */
export function formatTokens(tokens: number): string {
  if (tokens >= 1_000) {
    return `${(tokens / 1_000).toFixed(1)}k`;
  }
  return `${tokens}`;
}

/**
 * Whether a result is a single row of figures worth showing as metric cards.
 *
 * @param rows - The result rows.
 * @param columns - The result columns.
 * @returns True for a one-row result with at least one numeric column.
 */
export function isSingleValueResult(
  rows: ResultRow[],
  columns: string[],
): boolean {
  if (rows.length !== 1) {
    return false;
  }
  return columns.some((column) => typeof rows[0][column] === "number");
}
