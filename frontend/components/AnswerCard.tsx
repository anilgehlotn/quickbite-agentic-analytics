"use client";

/**
 * One complete answer.
 *
 * The visual hierarchy is deliberate and is the whole design of this card: the
 * headline is the answer and is set largest; the narrative explains it; the
 * findings, chart and table are evidence, in decreasing prominence; the caveats
 * are quiet but present. A reader who stops after the first line should have
 * been told the truth, and a reader who reads to the bottom should find nothing
 * that contradicts it.
 *
 * Caveats are styled as a calm aside rather than a warning. They are what a
 * careful analyst would say unprompted, and colouring them like an error would
 * teach a reader to skip them.
 */

import { useState } from "react";
import DataTable from "@/components/DataTable";
import ResultChart from "@/components/ResultChart";
import {
  formatCell,
  humaniseColumn,
  isMoneyColumn,
  isSingleValueResult,
} from "@/lib/format";
import type {
  AnalysisResponse,
  QueryResult,
  VerificationReport,
  VerificationStatus,
} from "@/lib/types";

/** Metric cards shown for a single-row result before the rest are dropped. */
const MAX_METRIC_CARDS = 4;

/** Colour and wording for each verification outcome. */
const VERIFICATION_TONE: Record<
  VerificationStatus,
  { label: string; className: string; dot: string }
> = {
  passed: {
    label: "Verified",
    className: "border-positive/30 bg-positive/10 text-positive",
    dot: "bg-positive",
  },
  passed_with_warnings: {
    label: "Verified with notes",
    className: "border-caution/30 bg-caution/10 text-caution",
    dot: "bg-caution",
  },
  failed: {
    label: "Unverified",
    className: "border-negative/30 bg-negative/10 text-negative",
    dot: "bg-negative",
  },
};

/**
 * The verification badge, expanding to show every individual check.
 *
 * @param props.report - The verifier's report.
 * @returns The rendered badge and its expandable check list.
 */
function VerificationBadge({
  report,
}: {
  report: VerificationReport;
}): JSX.Element {
  const [open, setOpen] = useState(false);
  const tone = VERIFICATION_TONE[report.status];
  const failed = report.checks.filter((check) => !check.passed);

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium transition-opacity hover:opacity-80 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${tone.className}`}
      >
        <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
        {tone.label}
        <span className="opacity-70">
          {report.checks.length - failed.length}/{report.checks.length} checks
        </span>
      </button>

      {open && (
        <div className="mt-3 space-y-2 rounded-lg border border-border bg-surface p-4">
          <p className="text-xs leading-relaxed text-muted">{report.summary}</p>
          <ul className="space-y-1.5 pt-1">
            {report.checks.map((check) => (
              <li key={check.name} className="flex gap-2 text-xs">
                <span
                  aria-hidden="true"
                  className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${
                    check.passed
                      ? "bg-positive/70"
                      : check.severity === "error"
                        ? "bg-negative"
                        : "bg-caution"
                  }`}
                />
                <span className="flex-1">
                  <span className="font-mono text-[0.6875rem] text-faint">
                    {check.name}
                  </span>
                  <span className="ml-2 text-muted">{check.message}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/**
 * Large single figures for a one-row result.
 *
 * @param props.result - The single-row result.
 * @returns The metric cards, or null when there are no numeric columns.
 */
function MetricCards({ result }: { result: QueryResult }): JSX.Element | null {
  const row = result.rows[0];
  const numeric = result.columns.filter(
    (column) => typeof row[column] === "number",
  );
  if (numeric.length === 0) {
    return null;
  }

  return (
    <dl className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {numeric.slice(0, MAX_METRIC_CARDS).map((column) => (
        <div
          key={column}
          className="rounded-lg border border-border bg-surface px-4 py-3.5"
        >
          <dt className="text-[0.6875rem] uppercase tracking-[0.1em] text-faint">
            {humaniseColumn(column)}
          </dt>
          <dd
            className={`mt-1.5 font-mono tabular-nums text-ink ${
              isMoneyColumn(column) ? "text-xl" : "text-2xl"
            }`}
          >
            {formatCell(column, row[column] ?? null)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Render one analysis, answered or not.
 *
 * @param props.response - The API response to render.
 * @param props.onRetry - Called when the reader asks to run the question again.
 * @returns The rendered card.
 */
export default function AnswerCard({
  response,
  onRetry,
}: {
  response: AnalysisResponse;
  onRetry?: (question: string) => void;
}): JSX.Element {
  const { insight, verification, query_results: results } = response;

  const singleValue = results.find(
    (result) =>
      result.error === null && isSingleValueResult(result.rows, result.columns),
  );

  if (!response.answered) {
    return (
      <article className="rounded-xl border border-border bg-surface p-6">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-caution" />
          <h3 className="text-sm font-medium text-ink">
            {insight?.headline ?? "This question could not be answered."}
          </h3>
        </div>

        <p className="mt-3 whitespace-pre-line text-sm leading-relaxed text-muted">
          {insight?.narrative ??
            response.error ??
            "The analysis did not complete and no reason was given."}
        </p>

        {insight !== null && response.error !== null && (
          <p className="mt-3 text-xs leading-relaxed text-faint">
            {response.error}
          </p>
        )}

        <div className="mt-5 flex flex-wrap items-center gap-3">
          {onRetry !== undefined && (
            <button
              type="button"
              onClick={() => onRetry(response.question)}
              className="rounded-lg border border-accent/40 bg-accent-soft px-3.5 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent hover:text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              Try again
            </button>
          )}
          {response.request_id !== null && (
            <span className="font-mono text-[0.6875rem] text-faint">
              request {response.request_id}
            </span>
          )}
        </div>
      </article>
    );
  }

  return (
    <article className="rounded-xl border border-border bg-surface p-6 sm:p-7">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {verification !== null && <VerificationBadge report={verification} />}
        {response.from_cache && (
          <span
            className="inline-flex items-center gap-1.5 rounded-full border border-border bg-raised px-2.5 py-1 text-[0.6875rem] text-faint"
            title="This answer was computed earlier and stored. Cached answers return instantly and stay available even when no model provider is reachable."
          >
            <span className="h-1 w-1 rounded-full bg-faint" />
            cached
          </span>
        )}
      </div>

      {insight !== null && (
        <>
          <h3 className="text-balance text-lg font-semibold leading-snug tracking-tight text-ink sm:text-xl">
            {insight.headline}
          </h3>

          <p className="mt-4 whitespace-pre-line text-sm leading-relaxed text-muted">
            {insight.narrative}
          </p>
        </>
      )}

      {singleValue !== undefined && <MetricCards result={singleValue} />}

      <ResultChart spec={response.chart} results={results} />

      {insight !== null && insight.key_findings.length > 0 && (
        <section className="mt-6">
          <h4 className="text-xs font-semibold uppercase tracking-[0.14em] text-faint">
            Key findings
          </h4>
          <ul className="mt-3 space-y-2">
            {insight.key_findings.map((finding, index) => (
              <li key={index} className="flex gap-3 text-sm leading-relaxed">
                <span
                  aria-hidden="true"
                  className="mt-2 h-1 w-1 shrink-0 rounded-full bg-accent"
                />
                <span className="text-muted">{finding}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {insight !== null && insight.recommended_actions.length > 0 && (
        <section className="mt-6">
          <h4 className="text-xs font-semibold uppercase tracking-[0.14em] text-faint">
            Recommended actions
          </h4>
          <ul className="mt-3 space-y-2">
            {insight.recommended_actions.map((action, index) => (
              <li
                key={index}
                className="rounded-lg border border-border bg-raised px-4 py-3 text-sm leading-relaxed text-muted"
              >
                {action}
              </li>
            ))}
          </ul>
        </section>
      )}

      {results
        .filter((result) => result.error === null && result.rows.length > 0)
        .map((result) => (
          <DataTable key={result.sub_query_id} result={result} />
        ))}

      {insight !== null && insight.caveats.length > 0 && (
        <section className="mt-6 rounded-lg border border-border/70 bg-base/40 px-4 py-3.5">
          <h4 className="text-[0.6875rem] font-semibold uppercase tracking-[0.14em] text-faint">
            Worth knowing
          </h4>
          <ul className="mt-2 space-y-1.5">
            {insight.caveats.map((caveat, index) => (
              <li key={index} className="text-xs leading-relaxed text-faint">
                {caveat}
              </li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}
