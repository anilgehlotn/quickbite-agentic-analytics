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
    className: "border-positive-line bg-positive-soft text-positive",
    dot: "bg-positive",
  },
  passed_with_warnings: {
    label: "Verified with notes",
    className: "border-caution-line bg-caution-soft text-caution",
    dot: "bg-caution",
  },
  failed: {
    label: "Unverified",
    className: "border-negative-line bg-negative-soft text-negative",
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
        className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-label font-semibold transition-colors hover:border-border-strong focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface ${tone.className}`}
      >
        <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
        {tone.label}
        <span className="font-normal tabular-nums text-muted">
          {report.checks.length - failed.length}/{report.checks.length} checks
        </span>
      </button>

      {open && (
        <div className="mt-3 rounded-lg border border-border bg-raised p-4">
          <p className="text-xs leading-relaxed text-muted">{report.summary}</p>
          <ul className="mt-3 space-y-2 border-t border-border pt-3">
            {report.checks.map((check) => (
              <li key={check.name} className="flex gap-2.5 text-xs">
                <span
                  aria-hidden="true"
                  className={`mt-[0.3125rem] h-1.5 w-1.5 shrink-0 rounded-full ${
                    check.passed
                      ? "bg-positive"
                      : check.severity === "error"
                        ? "bg-negative"
                        : "bg-caution"
                  }`}
                />
                <span className="min-w-0 flex-1">
                  <span className="font-mono text-micro text-faint">
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
    <dl className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {numeric.slice(0, MAX_METRIC_CARDS).map((column) => (
        <div
          key={column}
          className="rounded-lg border border-border bg-raised px-4 py-3.5"
        >
          <dt className="label-caps">{humaniseColumn(column)}</dt>
          <dd
            className={`mt-2 font-semibold tabular-nums tracking-tight text-ink ${
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
  onAsk,
  busy = false,
}: {
  response: AnalysisResponse;
  onRetry?: (question: string) => void;
  onAsk?: (question: string) => void;
  busy?: boolean;
}): JSX.Element {
  const { insight, verification, query_results: results } = response;
  const degraded = results.some((result) => result.degraded);

  const singleValue = results.find(
    (result) =>
      result.error === null && isSingleValueResult(result.rows, result.columns),
  );

  if (!response.answered) {
    return (
      <article className="rounded-lg border border-border bg-surface p-6">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-caution" />
          <span className="label-caps text-caution">Not answered</span>
        </div>

        <h3 className="mt-3 text-balance text-base font-semibold leading-snug tracking-tight text-ink">
          {insight?.headline ?? "This question could not be answered."}
        </h3>

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

        {/* A refusal that leaves the reader nowhere to go is only half an
            answer. These are complete questions the data can answer, adjacent
            to what was asked, submittable in one click. */}
        {response.suggested_questions.length > 0 && onAsk !== undefined && (
          <div className="mt-6">
            <h4 className="label-caps">Try one of these instead</h4>
            <ul className="mt-3 space-y-2">
              {response.suggested_questions.map((question) => (
                <li key={question}>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => onAsk(question)}
                    aria-label={`Ask: ${question}`}
                    className="group flex w-full items-start gap-3 rounded border border-border bg-canvas px-4 py-3 text-left text-sm leading-relaxed text-muted transition-colors hover:border-accent-line hover:bg-accent-soft hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <span
                      aria-hidden="true"
                      className="mt-[0.4375rem] h-1 w-1 shrink-0 rounded-full bg-accent"
                    />
                    <span>{question}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="mt-6 flex flex-wrap items-center gap-3">
          {onRetry !== undefined && (
            <button
              type="button"
              onClick={() => onRetry(response.question)}
              className="rounded border border-accent-line bg-accent-soft px-3.5 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent hover:text-accent-fg focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
            >
              Try again
            </button>
          )}
          {response.request_id !== null && (
            <span className="font-mono text-label text-faint">
              request {response.request_id}
            </span>
          )}
        </div>
      </article>
    );
  }

  return (
    <article className="rounded-lg border border-border bg-surface p-6 sm:p-8">
      <div className="mb-5 flex flex-wrap items-center gap-2">
        {verification !== null && <VerificationBadge report={verification} />}
        {response.from_cache && (
          <span
            className="inline-flex items-center gap-1.5 rounded-full border border-border bg-raised px-2.5 py-1 text-label text-faint"
            title="This answer was computed earlier and stored. Cached answers return instantly and stay available even when no model provider is reachable."
          >
            <span aria-hidden="true" className="h-1 w-1 rounded-full bg-faint" />
            cached
          </span>
        )}
        {degraded && (
          <span
            className="inline-flex items-center gap-1.5 rounded-full border border-caution-line bg-caution-soft px-2.5 py-1 text-label text-caution"
            title="No language model was reachable, so the SQL was assembled directly from the plan. The figures are exact — they come from the same database — but the query is a plain aggregate rather than one tailored to the question."
          >
            <span aria-hidden="true" className="h-1 w-1 rounded-full bg-caution" />
            simplified query
          </span>
        )}
      </div>

      {insight !== null && (
        <>
          <h3 className="text-balance text-xl font-semibold leading-[1.3] tracking-tight text-ink sm:text-2xl">
            {insight.headline}
          </h3>

          <p className="mt-5 whitespace-pre-line text-[0.9375rem] leading-[1.7] text-muted">
            {insight.narrative}
          </p>
        </>
      )}

      {singleValue !== undefined && <MetricCards result={singleValue} />}

      <ResultChart spec={response.chart} results={results} />

      {insight !== null && insight.key_findings.length > 0 && (
        <section className="mt-8">
          <h4 className="label-caps">Key findings</h4>
          <ul className="mt-3.5 space-y-2.5">
            {insight.key_findings.map((finding, index) => (
              <li key={index} className="flex gap-3 text-sm leading-relaxed">
                <span
                  aria-hidden="true"
                  className="mt-[0.5625rem] h-1 w-1 shrink-0 rounded-full bg-accent"
                />
                <span className="text-muted">{finding}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {insight !== null && insight.recommended_actions.length > 0 && (
        <section className="mt-8">
          <h4 className="label-caps">Recommended actions</h4>
          <ul className="mt-3.5 space-y-2">
            {insight.recommended_actions.map((action, index) => (
              <li
                key={index}
                className="border-l-2 border-accent-line bg-raised px-4 py-3 text-sm leading-relaxed text-muted"
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
        <section className="mt-8 rounded-lg bg-raised px-4 py-4">
          <h4 className="label-caps">Worth knowing</h4>
          <ul className="mt-2.5 space-y-1.5">
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
