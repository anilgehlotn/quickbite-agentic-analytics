"use client";

/**
 * The agent trace: what ran, in what order, how long it took and what it
 * decided.
 *
 * This panel is the evidence for the system's central claim. "Four agents
 * collaborated on this answer" is unfalsifiable prose; a trace showing the
 * planner's decision, the SQL the analyst actually executed, the verifier's
 * check count and the token spend is something a reader can audit. So failed
 * and skipped steps are rendered exactly like successful ones — a visible retry
 * that recovered earns more trust than an unbroken row of ticks, and hiding a
 * failure here would undermine the only reason the panel exists.
 *
 * While a request is in flight the panel shows the pipeline advancing. Those
 * in-flight steps deliberately carry no duration and no token count: the timing
 * is a schedule, not a measurement, and inventing numbers in the one component
 * whose job is to be trustworthy would be self-defeating. Every figure shown
 * after the response lands is real.
 */

import { useCallback, useState } from "react";
import { formatDuration, formatTokens } from "@/lib/format";
import type {
  AgentStatus,
  AgentStep,
  AnalysisResponse,
  QueryResult,
} from "@/lib/types";

/** The pipeline's stages, in execution order, with reader-facing names. */
const STAGES: ReadonlyArray<{
  key: string;
  label: string;
  description: string;
  /** Elapsed milliseconds after which this stage is shown as running. */
  startsAt: number;
}> = [
  {
    key: "planner",
    label: "Planner",
    description: "Interpreting the question and choosing the analysis",
    startsAt: 0,
  },
  {
    key: "sql_analyst",
    label: "SQL analyst",
    description: "Writing and executing the queries",
    startsAt: 8_000,
  },
  {
    key: "verifier",
    label: "Verifier",
    description: "Checking the numbers against each other",
    startsAt: 14_000,
  },
  {
    key: "insight",
    label: "Insight",
    description: "Explaining what the result means",
    startsAt: 16_000,
  },
];

/** Dot colour and label for each step status. */
const STATUS_STYLE: Record<
  AgentStatus,
  { dot: string; label: string; text: string }
> = {
  pending: { dot: "bg-border", label: "waiting", text: "text-faint" },
  running: {
    dot: "bg-accent animate-pulse-soft",
    label: "running",
    text: "text-accent",
  },
  succeeded: { dot: "bg-positive", label: "done", text: "text-muted" },
  failed: { dot: "bg-negative", label: "failed", text: "text-negative" },
  skipped: { dot: "bg-faint", label: "skipped", text: "text-faint" },
};

/**
 * Turn an agent name into a reader-facing label.
 *
 * @param name - The agent name from the trace, e.g. "sql_analyst_retry".
 * @returns A display label, e.g. "SQL analyst (retry)".
 */
function stepLabel(name: string): string {
  const retry = name.endsWith("_retry");
  const base = retry ? name.slice(0, -"_retry".length) : name;
  const stage = STAGES.find((entry) => entry.key === base);
  const label =
    stage?.label ??
    base
      .split("_")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  return retry ? `${label} (retry)` : label;
}

/**
 * A copy-to-clipboard button that reports what it did.
 *
 * @param props.value - The text to copy.
 * @param props.label - Accessible label for the control.
 * @returns The rendered button.
 */
function CopyButton({
  value,
  label,
}: {
  value: string;
  label: string;
}): JSX.Element {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1_600);
    } catch {
      // Clipboard access can be denied; the SQL is selectable either way.
      setCopied(false);
    }
  }, [value]);

  return (
    <button
      type="button"
      onClick={() => void copy()}
      aria-label={label}
      className="rounded-sm border border-border px-1.5 py-0.5 text-micro text-faint transition-colors hover:border-accent-line hover:bg-accent-soft hover:text-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
    >
      {copied ? "copied" : "copy"}
    </button>
  );
}

/**
 * One step in the pipeline.
 *
 * @param props.name - Agent name.
 * @param props.status - Execution state.
 * @param props.summary - What the step decided.
 * @param props.durationMs - How long it took, when measured.
 * @param props.provider - Which provider served it, when any.
 * @param props.tokens - Tokens consumed, when any.
 * @param props.error - Failure reason, when it failed.
 * @param props.last - Whether to omit the connecting line below.
 * @returns The rendered step.
 */
function Step({
  name,
  status,
  summary,
  durationMs,
  provider,
  tokens,
  error,
  last,
  statusLabel,
}: {
  name: string;
  status: AgentStatus;
  summary: string;
  durationMs?: number;
  provider?: string | null;
  tokens?: number | null;
  error?: string | null;
  last: boolean;
  statusLabel?: string;
}): JSX.Element {
  const style = STATUS_STYLE[status];

  return (
    <li className="relative flex gap-3 pb-5 last:pb-0">
      {/* The hairline between dots is what makes the steps read as a pipeline
          rather than an unordered list. */}
      {!last && (
        <span
          aria-hidden="true"
          className="absolute left-[3px] top-3 h-full w-px bg-border"
        />
      )}
      <span
        aria-hidden="true"
        className={`relative mt-[0.3125rem] h-[7px] w-[7px] shrink-0 rounded-full ${style.dot}`}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-3">
          <span className="text-xs font-semibold tracking-tight text-ink">
            {stepLabel(name)}
          </span>
          <span
            className={`shrink-0 text-label tabular-nums ${style.text}`}
          >
            {durationMs !== undefined
              ? formatDuration(durationMs)
              : (statusLabel ?? style.label)}
          </span>
        </div>

        <p className="mt-1 text-label leading-relaxed text-muted">{summary}</p>

        {error !== undefined && error !== null && (
          <p className="mt-1 text-label leading-relaxed text-negative">
            {error}
          </p>
        )}

        {(provider ?? null) !== null && (
          <p className="mt-1.5 font-mono text-micro tabular-nums text-faint">
            {provider}
            {tokens !== undefined && tokens !== null
              ? ` · ${formatTokens(tokens)} tokens`
              : ""}
          </p>
        )}
      </div>
    </li>
  );
}

/**
 * The executed SQL for one sub-query.
 *
 * @param props.result - The result carrying the SQL and its outcome.
 * @returns The rendered block.
 */
function SqlBlock({ result }: { result: QueryResult }): JSX.Element {
  return (
    <div className="min-w-0 overflow-hidden rounded border border-border bg-raised">
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-1.5">
        <span className="truncate font-mono text-micro text-muted">
          {result.sub_query_id}
        </span>
        <div className="flex shrink-0 items-center gap-2">
          <span className="font-mono text-micro tabular-nums text-faint">
            {result.error !== null
              ? "failed"
              : `${result.row_count.toLocaleString("en-IN")} rows · ${formatDuration(
                  result.execution_ms,
                )}`}
          </span>
          {result.attempts > 1 && (
            <span
              className="rounded-sm border border-caution-line bg-caution-soft px-1.5 py-px text-micro text-caution"
              title="The first attempt failed and was repaired using the database's own error message"
            >
              retry {result.attempts}
            </span>
          )}
          {result.sql.length > 0 && (
            <CopyButton
              value={result.sql}
              label={`Copy SQL for ${result.sub_query_id}`}
            />
          )}
        </div>
      </div>
      {result.error !== null ? (
        <p className="bg-negative-soft px-3 py-2 text-label leading-relaxed text-negative">
          {result.error}
        </p>
      ) : (
        <pre className="max-h-56 overflow-auto px-3 py-2.5 font-mono text-label leading-[1.65] text-muted">
          <code>{result.sql}</code>
        </pre>
      )}
    </div>
  );
}

/**
 * The trace panel: live pipeline while running, full record afterwards.
 *
 * @param props.response - The completed analysis, when there is one.
 * @param props.busy - Whether a request is currently in flight.
 * @param props.elapsedMs - Milliseconds since the request started.
 * @param props.hideHeader - Omit the panel's own header, used when it is
 *   nested inside a disclosure that already labels it.
 * @returns The rendered panel.
 */
export default function TracePanel({
  response,
  busy,
  elapsedMs,
  hideHeader = false,
}: {
  response: AnalysisResponse | null;
  busy: boolean;
  elapsedMs: number;
  hideHeader?: boolean;
}): JSX.Element {
  const [showSql, setShowSql] = useState(true);

  const steps: AgentStep[] = response?.trace.steps ?? [];
  const plan = response?.plan ?? null;

  return (
    <section
      aria-label="Agent execution trace"
      className={
        hideHeader
          ? "min-w-0"
          : "min-w-0 rounded-lg border border-border bg-surface"
      }
    >
      {!hideHeader && (
        <header className="flex items-center justify-between gap-3 border-b border-border px-5 py-3.5">
          <h2 className="label-caps">Agent trace</h2>
          {busy ? (
            <span className="inline-flex items-center gap-2 text-label text-accent">
              <span
                aria-hidden="true"
                className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-accent"
              />
              running
            </span>
          ) : (
            response !== null && (
              <span className="text-label tabular-nums text-faint">
                {formatDuration(response.trace.total_duration_ms)} ·{" "}
                {formatTokens(response.trace.total_tokens)} tokens
              </span>
            )
          )}
        </header>
      )}

      <div className={hideHeader ? "" : "px-5 py-5"}>
        {busy ? (
          <>
            <ol className="mt-1">
              {STAGES.map((stage, index) => {
                const next = STAGES[index + 1];
                const started = elapsedMs >= stage.startsAt;
                const finished =
                  next !== undefined && elapsedMs >= next.startsAt;
                const status: AgentStatus = finished
                  ? "succeeded"
                  : started
                    ? "running"
                    : "pending";
                return (
                  <Step
                    key={stage.key}
                    name={stage.key}
                    status={status}
                    summary={stage.description}
                    last={index === STAGES.length - 1}
                    // Deliberately not "done": the stage order is certain but
                    // the timing is a schedule, and this panel does not get to
                    // assert things it has not measured.
                    statusLabel={finished ? "·" : undefined}
                  />
                );
              })}
            </ol>
            <p className="mt-2 border-t border-border pt-3 text-micro leading-relaxed text-faint">
              Stage order is fixed; the highlight is indicative until the run
              finishes. Measured durations and token counts appear below when it
              does.
            </p>
          </>
        ) : response === null ? (
          <div className="py-2">
            <p className="text-xs leading-relaxed text-muted">
              Four agents answer every question: a planner interprets it, a SQL
              analyst writes and runs the queries, a verifier checks the numbers
              arithmetically, and an insight agent explains the result.
            </p>
            <p className="mt-3 text-xs leading-relaxed text-faint">
              Their steps, the exact SQL executed, the duration and the token
              cost all appear here once you ask something.
            </p>
            <ol className="mt-6">
              {STAGES.map((stage, index) => (
                <Step
                  key={stage.key}
                  name={stage.key}
                  status="pending"
                  summary={stage.description}
                  last={index === STAGES.length - 1}
                />
              ))}
            </ol>
          </div>
        ) : (
          <>
            <ol>
              {steps.map((step, index) => (
                <Step
                  key={`${step.agent_name}-${index}`}
                  name={step.agent_name}
                  status={step.status}
                  summary={step.summary}
                  durationMs={step.duration_ms}
                  provider={step.llm_provider}
                  tokens={step.tokens}
                  error={step.error}
                  last={index === steps.length - 1}
                />
              ))}
            </ol>

            {plan !== null && (
              <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-2 rounded border border-border bg-raised px-4 py-3.5 text-label">
                <dt className="text-faint">Intent</dt>
                <dd className="text-right font-mono text-muted">
                  {plan.intent}
                </dd>
                <dt className="text-faint">Window</dt>
                <dd className="text-right font-mono tabular-nums text-muted">
                  {plan.time_window.start_date} → {plan.time_window.end_date}
                </dd>
                {plan.time_window.comparison_start !== null && (
                  <>
                    <dt className="text-faint">Baseline</dt>
                    <dd className="text-right font-mono tabular-nums text-muted">
                      {plan.time_window.comparison_start} →{" "}
                      {plan.time_window.comparison_end}
                    </dd>
                  </>
                )}
                <dt className="text-faint">Metrics</dt>
                <dd className="text-right font-mono text-muted">
                  {plan.metrics.length > 0 ? plan.metrics.join(", ") : "—"}
                </dd>
                <dt className="text-faint">Sub-queries</dt>
                <dd className="text-right font-mono tabular-nums text-muted">
                  {plan.sub_queries.length}
                </dd>
                {response.trace.providers_used.length > 0 && (
                  <>
                    <dt className="text-faint">Provider</dt>
                    <dd className="text-right font-mono text-muted">
                      {response.trace.providers_used.join(", ")}
                    </dd>
                  </>
                )}
              </dl>
            )}

            {response.query_results.length > 0 && (
              <div className="mt-6 border-t border-border pt-4">
                <button
                  type="button"
                  onClick={() => setShowSql((current) => !current)}
                  aria-expanded={showSql}
                  className="group flex w-full items-center justify-between gap-3 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  <span className="label-caps">Executed SQL</span>
                  <span className="text-label text-faint group-hover:text-accent">
                    {showSql ? "Hide" : "Show"}
                  </span>
                </button>

                {showSql && (
                  <div className="mt-3 space-y-2">
                    {response.query_results.map((result) => (
                      <SqlBlock key={result.sub_query_id} result={result} />
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
