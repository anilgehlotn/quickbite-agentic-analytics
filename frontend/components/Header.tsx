"use client";

/**
 * The application header: identity, the data's as-of date, and backend status.
 *
 * The as-of date is in the header rather than a footnote because it changes the
 * meaning of every answer on the page. This dataset ends on 2026-07-31, so
 * "last 3 months" means May to July 2026 and not the three months before today;
 * a reader who does not know that will misread every figure.
 */

import type { HealthResponse } from "@/lib/types";

/**
 * The header bar.
 *
 * @param props.health - The backend health payload, when it has answered.
 * @param props.checking - Whether the health probe is still in flight.
 * @returns The rendered header.
 */
export default function Header({
  health,
  checking,
}: {
  health: HealthResponse | null;
  checking: boolean;
}): JSX.Element {
  // The pill reports what the service can do, not merely whether it answered.
  // Saying "connected" while a banner two inches below explains that live
  // analysis is unavailable would be two truths that read as a contradiction.
  const tone = checking
    ? { dot: "bg-caution animate-pulse-soft", text: "text-muted", label: "connecting" }
    : health === null
      ? { dot: "bg-negative", text: "text-negative", label: "offline" }
      : health.mode === "full"
        ? { dot: "bg-positive", text: "text-muted", label: "connected" }
        : health.mode === "cache_only"
          ? { dot: "bg-caution", text: "text-caution", label: "cached answers only" }
          : { dot: "bg-negative", text: "text-negative", label: "unavailable" };

  const title =
    health === null
      ? "The backend has not responded."
      : `mode ${health.mode} · ${health.providers_configured.length} provider(s) ` +
        `configured · ${health.cached_answers} cached answers · v${health.version}`;

  return (
    <header className="sticky top-0 z-20 border-b border-border bg-canvas">
      <div className="mx-auto flex max-w-[92rem] items-center justify-between gap-4 px-5 py-3.5 sm:px-8">
        <div className="flex items-center gap-3">
          <span className="text-[0.9375rem] font-semibold tracking-tight text-ink">
            QuickBite
          </span>
          <span aria-hidden="true" className="h-3.5 w-px bg-border" />
          <span className="text-xs tracking-tight text-faint">
            Agentic analytics
          </span>
        </div>

        <div className="flex items-center gap-3">
          {health !== null && (
            <span className="hidden text-label tabular-nums text-faint sm:inline">
              data as of {health.data_asof}
            </span>
          )}
          <span
            title={title}
            className={`inline-flex items-center gap-2 rounded-full border border-border bg-surface px-2.5 py-1 text-label ${tone.text}`}
          >
            <span
              aria-hidden="true"
              className={`h-1.5 w-1.5 rounded-full ${tone.dot}`}
            />
            {tone.label}
          </span>
        </div>
      </div>
    </header>
  );
}
