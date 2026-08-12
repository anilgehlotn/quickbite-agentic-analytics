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
    <div className="mt-4 overflow-hidden rounded-lg border border-border">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 bg-surface px-4 py-2.5 text-left text-xs text-muted transition-colors hover:bg-raised focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <span>
          <span className="font-medium text-ink">Data</span>
          <span className="ml-2 text-faint">
            {result.row_count.toLocaleString("en-IN")}{" "}
            {result.row_count === 1 ? "row" : "rows"} · {result.columns.length}{" "}
            {result.columns.length === 1 ? "column" : "columns"}
          </span>
        </span>
        <span className="text-faint">{open ? "Hide" : "Show"}</span>
      </button>

      {open && (
        <div className="overflow-x-auto border-t border-border">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="bg-raised">
                {result.columns.map((column) => (
                  <th
                    key={column}
                    scope="col"
                    className="whitespace-nowrap px-3 py-2 text-left font-medium text-muted"
                  >
                    {humaniseColumn(column)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr
                  key={index}
                  className="border-t border-border/60 even:bg-surface/40"
                >
                  {result.columns.map((column) => {
                    const value = row[column];
                    const numeric = typeof value === "number";
                    return (
                      <td
                        key={column}
                        className={`whitespace-nowrap px-3 py-1.5 ${
                          numeric
                            ? "text-right font-mono tabular-nums text-ink"
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
              className="w-full border-t border-border bg-surface px-4 py-2 text-xs text-accent transition-colors hover:bg-raised focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
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
