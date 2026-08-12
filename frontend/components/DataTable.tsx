"use client";

/**
 * The raw rows behind an answer, collapsed by default.
 *
 * Collapsed because the narrative is the answer and the table is the evidence;
 * present because a reader who doubts a figure should be able to see the row it
 * came from without leaving the page. Numeric columns are right-aligned with
 * tabular numerals so digits line up and magnitudes can be compared by eye.
 */

import { useState } from "react";
import { formatCell, humaniseColumn } from "@/lib/format";
import type { QueryResult } from "@/lib/types";

/** Rows shown before the table offers to stop. */
const PREVIEW_ROWS = 25;

/**
 * A collapsible table of one query result.
 *
 * @param props.result - The result to render.
 * @param props.defaultOpen - Whether to start expanded.
 * @returns The rendered table section.
 */
export default function DataTable({
  result,
  defaultOpen = false,
}: {
  result: QueryResult;
  defaultOpen?: boolean;
}): JSX.Element | null {
  const [open, setOpen] = useState(defaultOpen);
  const [showAll, setShowAll] = useState(false);

  if (result.error !== null || result.rows.length === 0) {
    return null;
  }

  const rows = showAll ? result.rows : result.rows.slice(0, PREVIEW_ROWS);
  const hidden = result.rows.length - rows.length;

  return (
    <div className="mt-4 overflow-hidden rounded border border-border">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        className="group flex w-full items-center justify-between gap-3 bg-raised px-4 py-2.5 text-left text-xs transition-colors hover:bg-sunken focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <span>
          <span className="font-semibold text-ink">Data</span>
          <span className="ml-2 tabular-nums text-faint">
            {result.row_count.toLocaleString("en-IN")}{" "}
            {result.row_count === 1 ? "row" : "rows"} · {result.columns.length}{" "}
            {result.columns.length === 1 ? "column" : "columns"}
          </span>
        </span>
        <span className="text-label text-faint group-hover:text-accent">
          {open ? "Hide" : "Show"}
        </span>
      </button>

      {open && (
        <div className="overflow-x-auto border-t border-border bg-surface">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="border-b border-border">
                {result.columns.map((column) => {
                  // Right-align the header of a numeric column so it sits over
                  // its own digits rather than over the column to its left.
                  const numeric = rows.some(
                    (row) => typeof row[column] === "number",
                  );
                  return (
                    <th
                      key={column}
                      scope="col"
                      className={`label-caps whitespace-nowrap px-3 py-2.5 ${
                        numeric ? "text-right" : "text-left"
                      }`}
                    >
                      {humaniseColumn(column)}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index} className="border-b border-border last:border-b-0">
                  {result.columns.map((column) => {
                    const value = row[column];
                    const numeric = typeof value === "number";
                    return (
                      <td
                        key={column}
                        className={`whitespace-nowrap px-3 py-2 ${
                          numeric
                            ? "text-right tabular-nums text-ink"
                            : "text-muted"
                        }`}
                      >
                        {formatCell(column, value ?? null)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>

          {hidden > 0 && (
            <button
              type="button"
              onClick={() => setShowAll(true)}
              className="w-full border-t border-border bg-raised px-4 py-2 text-xs font-medium text-accent transition-colors hover:bg-accent-soft focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              Show {hidden.toLocaleString("en-IN")} more{" "}
              {hidden === 1 ? "row" : "rows"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
