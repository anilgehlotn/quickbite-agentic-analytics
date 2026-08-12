"use client";

/**
 * The banner shown when the service cannot answer new questions.
 *
 * Placed directly above the input rather than in a corner, because its job is
 * to be read *before* someone types. Letting a reviewer compose a question into
 * a box that cannot answer it, and only then explaining why, would waste the
 * one interaction they were willing to give.
 *
 * The wording leads with what still works. In cache-only mode the data layer,
 * the semantic layer, the SQL guard and the verification checks are all
 * healthy, and eight complete analyses are one click away; only new questions
 * are unavailable. Describing that as an outage would be inaccurate.
 */

import type { ServiceMode } from "@/lib/types";

/**
 * A short, honest statement of what the service can currently do.
 *
 * @param props.mode - The service mode reported by the health endpoint.
 * @param props.cachedAnswers - How many prepared analyses are available.
 * @returns The banner, or null when the service is fully operational.
 */
export default function ModeNotice({
  mode,
  cachedAnswers,
}: {
  mode: ServiceMode;
  cachedAnswers: number;
}): JSX.Element | null {
  if (mode === "full") {
    return null;
  }

  if (mode === "offline") {
    return (
      <div
        role="status"
        className="mb-3 rounded-lg border border-negative/30 bg-negative/5 px-4 py-3"
      >
        <p className="text-xs font-medium text-negative">
          The analytics service is not reachable.
        </p>
        <p className="mt-1.5 text-xs leading-relaxed text-muted">
          Nothing can be answered until it responds. If this is the deployed
          demo, the backend may be starting up — it suspends when idle and can
          take up to a minute to wake. Try again shortly.
        </p>
      </div>
    );
  }

  return (
    <div
      role="status"
      className="mb-3 rounded-lg border border-caution/30 bg-caution/5 px-4 py-3"
    >
      <p className="text-xs font-medium text-caution">
        Live analysis is unavailable right now — {cachedAnswers} prepared
        analyses are ready.
      </p>
      <p className="mt-1.5 text-xs leading-relaxed text-muted">
        No language model provider is currently reachable, so a new question
        cannot be planned or queried. The database, semantic layer and
        verification checks are all working: the questions below were computed
        earlier by the full agent pipeline and return instantly, each with its
        complete trace, executed SQL and verification report.
      </p>
    </div>
  );
}
