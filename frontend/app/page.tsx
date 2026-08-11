"use client";

import { useCallback, useEffect, useState } from "react";
import {
  API_URL,
  ApiError,
  COLD_START_HINT_MS,
  getHealth,
  type HealthResponse,
} from "@/lib/api";

type Phase = "loading" | "ready" | "error";

/**
 * Format an integer with thousands separators.
 *
 * @param value - The number to format, or null when unavailable.
 * @returns The formatted number, or an em dash when there is nothing to show.
 */
function formatCount(value: number | null): string {
  return value === null ? "—" : value.toLocaleString("en-IN");
}

/**
 * A labelled figure in the status card.
 *
 * @param props.label - Field name.
 * @param props.value - Field value, already formatted.
 * @param props.mono - Whether to render the value in a monospace face.
 * @returns The rendered row.
 */
function Field({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}): JSX.Element {
  return (
    <div className="flex items-baseline justify-between gap-6 border-b border-border/60 py-3 last:border-0">
      <dt className="text-sm text-muted">{label}</dt>
      <dd
        className={`text-sm text-ink ${mono ? "font-mono text-[0.8125rem]" : ""}`}
      >
        {value}
      </dd>
    </div>
  );
}

/**
 * A coloured status dot with a label.
 *
 * @param props.tone - Which semantic colour to use.
 * @param props.label - Text beside the dot.
 * @param props.pulsing - Whether the dot should pulse, used while loading.
 * @returns The rendered indicator.
 */
function StatusPill({
  tone,
  label,
  pulsing = false,
}: {
  tone: "positive" | "caution" | "negative";
  label: string;
  pulsing?: boolean;
}): JSX.Element {
  const dot = {
    positive: "bg-positive",
    caution: "bg-caution",
    negative: "bg-negative",
  }[tone];

  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-border bg-raised px-3 py-1 text-xs font-medium text-ink">
      <span
        className={`h-1.5 w-1.5 rounded-full ${dot} ${
          pulsing ? "animate-pulse-soft" : ""
        }`}
      />
      {label}
    </span>
  );
}

/**
 * Landing page with a live backend status card.
 *
 * Fetches /health on mount. The three states are distinct and honest: while
 * loading it says so and, past a few seconds, explains that a free-tier backend
 * may be waking; on failure it reports the actual reason and offers a retry;
 * it never renders a healthy-looking card when the backend did not answer.
 *
 * @returns The rendered page.
 */
export default function Home(): JSX.Element {
  const [phase, setPhase] = useState<Phase>("loading");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [slow, setSlow] = useState(false);

  const load = useCallback(async () => {
    setPhase("loading");
    setError(null);
    setSlow(false);

    const hint = setTimeout(() => setSlow(true), COLD_START_HINT_MS);
    try {
      setHealth(await getHealth());
      setPhase("ready");
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : "An unexpected error occurred while contacting the API.",
      );
      setPhase("error");
    } finally {
      clearTimeout(hint);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const degraded = health?.status === "degraded";

  return (
    <main className="page-glow min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-content flex-col px-6 py-16 sm:px-8 lg:py-24">
        <header className="max-w-2xl">
          <span className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium tracking-wide text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />
            QSR analytics
          </span>

          <h1 className="mt-7 text-4xl font-semibold tracking-tight text-ink sm:text-display">
            QuickBite<span className="text-accent">.</span>
          </h1>

          <p className="mt-5 text-lede text-muted">
            Natural-language business analytics powered by a multi-agent AI
            system.
          </p>

          <p className="mt-4 max-w-xl text-sm leading-relaxed text-faint">
            Ask a question in plain English. Agents plan the analysis, write and
            run the SQL, verify the numbers against a semantic layer, and explain
            what the result means for the business.
          </p>
        </header>

        <section className="mt-14" aria-labelledby="status-heading">
          <div className="flex items-center justify-between gap-4">
            <h2
              id="status-heading"
              className="text-xs font-semibold uppercase tracking-[0.14em] text-faint"
            >
              System status
            </h2>
            {phase === "ready" ? (
              <StatusPill
                tone={degraded ? "caution" : "positive"}
                label={degraded ? "Degraded" : "Connected"}
              />
            ) : phase === "loading" ? (
              <StatusPill tone="caution" label="Connecting" pulsing />
            ) : (
              <StatusPill tone="negative" label="Unreachable" />
            )}
          </div>

          <div className="mt-4 rounded-xl border border-border bg-surface p-6 sm:p-7">
            {phase === "loading" && (
              <div>
                <p className="text-sm text-muted">
                  Contacting the analytics API…
                </p>
                {slow && (
                  <p className="mt-3 max-w-lg text-sm leading-relaxed text-faint">
                    This is taking a moment. The backend runs on a free tier that
                    suspends idle services, so a cold start can take up to 50
                    seconds. The request has not failed — it is still waiting.
                  </p>
                )}
              </div>
            )}

            {phase === "error" && (
              <div>
                <p className="text-sm font-medium text-negative">
                  Could not reach the backend.
                </p>
                <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">
                  {error}
                </p>
                <p className="mt-3 font-mono text-xs text-faint">{API_URL}</p>
                <button
                  type="button"
                  onClick={() => void load()}
                  className="mt-5 rounded-lg border border-accent/40 bg-accent-soft px-4 py-2 text-sm font-medium text-accent transition-colors hover:bg-accent hover:text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  Try again
                </button>
              </div>
            )}

            {phase === "ready" && health && (
              <div>
                {degraded && (
                  <p className="mb-5 rounded-lg border border-caution/30 bg-caution/10 px-4 py-3 text-sm text-caution">
                    The API is running but its database is not readable.
                    {health.database_error ? ` ${health.database_error}` : ""}
                  </p>
                )}

                <dl>
                  <Field
                    label="Connection"
                    value={degraded ? "Reachable, degraded" : "Healthy"}
                  />
                  <Field
                    label="Orders in database"
                    value={formatCount(health.fact_orders_rows)}
                    mono
                  />
                  <Field label="Data as of" value={health.data_asof} mono />
                  <Field
                    label="LLM providers"
                    value={
                      health.providers_configured.length > 0
                        ? health.providers_configured.join(", ")
                        : "none configured"
                    }
                  />
                  <Field label="Environment" value={health.environment} />
                  <Field label="API version" value={health.version} mono />
                </dl>
              </div>
            )}
          </div>
        </section>

        <footer className="mt-auto pt-16">
          <p className="text-xs text-faint">
            Step 2.1 — deployment skeleton. Query interface and agents arrive in
            a later step.
          </p>
        </footer>
      </div>
    </main>
  );
}
