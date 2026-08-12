"use client";

/**
 * The application: a conversation on the left, the agent trace on the right.
 *
 * The two-column layout is the argument the page makes. Most analytics demos
 * show an answer and ask to be believed; putting the trace permanently beside
 * the answer means the evidence is never more than a glance away, and the wait
 * during a live run becomes the most interesting part of the interface rather
 * than dead time.
 *
 * On narrow screens the trace collapses beneath each answer, because a sticky
 * side panel on a phone is a scroll trap.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import AnswerCard from "@/components/AnswerCard";
import Header from "@/components/Header";
import QuestionInput from "@/components/QuestionInput";
import SuggestionChips from "@/components/SuggestionChips";
import TracePanel from "@/components/TracePanel";
import {
  ANALYSIS_COLD_START_HINT_MS,
  API_URL,
  ApiError,
  askQuestion,
  getCanonicalQuestions,
  getHealth,
} from "@/lib/api";
import type {
  AnalysisResponse,
  HealthResponse,
  QuestionSuggestion,
} from "@/lib/types";

/** How often the in-flight elapsed timer ticks, in milliseconds. */
const TICK_MS = 250;

/** One entry in the conversation. */
interface Turn {
  /** Stable key for React. */
  id: number;
  /** The question as asked. */
  question: string;
  /** The answer, once it arrives. */
  response: AnalysisResponse | null;
  /** The failure, when the request could not complete. */
  error: ApiError | null;
}

/**
 * The failure state for a request that never reached the pipeline.
 *
 * A transport failure is different from an unanswerable question: there is no
 * trace, no plan and nothing to show but the reason and a way to try again.
 *
 * @param props.error - What went wrong.
 * @param props.question - The question that failed, for the retry.
 * @param props.onRetry - Called to run the question again.
 * @returns The rendered error card.
 */
function ErrorCard({
  error,
  question,
  onRetry,
}: {
  error: ApiError;
  question: string;
  onRetry: (question: string) => void;
}): JSX.Element {
  const rateLimited = error.isRateLimited;

  return (
    <article
      className={`rounded-xl border p-6 ${
        rateLimited
          ? "border-caution/30 bg-caution/5"
          : "border-negative/30 bg-negative/5"
      }`}
      role="alert"
    >
      <h3
        className={`text-sm font-medium ${
          rateLimited ? "text-caution" : "text-negative"
        }`}
      >
        {rateLimited ? "Rate limit reached" : "The request did not complete"}
      </h3>

      <p className="mt-2 text-sm leading-relaxed text-muted">{error.message}</p>

      {rateLimited && error.retryAfter !== undefined && (
        <p className="mt-2 text-sm leading-relaxed text-muted">
          You can ask again in about {Math.ceil(error.retryAfter / 60) || 1}{" "}
          minute
          {Math.ceil(error.retryAfter / 60) > 1 ? "s" : ""}. The suggested
          questions below are cached and remain available in the meantime.
        </p>
      )}

      {!rateLimited && (
        <p className="mt-2 text-xs leading-relaxed text-faint">
          Backend: <span className="font-mono">{API_URL}</span>
        </p>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => onRetry(question)}
          className="rounded-lg border border-accent/40 bg-accent-soft px-3.5 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent hover:text-base focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          Try again
        </button>
        {error.requestId !== undefined && (
          <span className="font-mono text-[0.6875rem] text-faint">
            request {error.requestId}
          </span>
        )}
      </div>
    </article>
  );
}

/**
 * The empty state, shown before the first question.
 *
 * @param props.hasSuggestions - Whether the chips loaded, which decides
 *   whether it is honest to tell the reader to pick one.
 * @returns The rendered introduction.
 */
function EmptyState({
  hasSuggestions,
}: {
  hasSuggestions: boolean;
}): JSX.Element {
  return (
    <div className="rounded-xl border border-border bg-surface p-6 sm:p-8">
      <h1 className="text-xl font-semibold tracking-tight text-ink sm:text-2xl">
        Ask a business question in plain English.
      </h1>
      <p className="mt-4 max-w-xl text-sm leading-relaxed text-muted">
        Four agents answer it: a <strong className="text-ink">planner</strong>{" "}
        works out what to measure, a{" "}
        <strong className="text-ink">SQL analyst</strong> writes and runs the
        queries, a <strong className="text-ink">verifier</strong> checks the
        numbers arithmetically before you see them, and an{" "}
        <strong className="text-ink">insight agent</strong> explains what the
        result means and what to do about it.
      </p>
      <p className="mt-3 max-w-xl text-sm leading-relaxed text-faint">
        Every step, including the exact SQL, appears in the trace panel.{" "}
        {hasSuggestions
          ? "Pick a question below to start."
          : "Type a question below to start."}
      </p>
    </div>
  );
}

/**
 * The main application page.
 *
 * @returns The rendered application.
 */
export default function Home(): JSX.Element {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [suggestions, setSuggestions] = useState<QuestionSuggestion[]>([]);
  const [loadingSuggestions, setLoadingSuggestions] = useState(true);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(true);

  const nextId = useRef(0);
  const latestTurnRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const payload = await getCanonicalQuestions();
        if (!cancelled) {
          setSuggestions(payload.questions);
        }
      } catch {
        // The chips are an accelerator, not a requirement; the input still
        // works without them and the header will show the backend as offline.
        if (!cancelled) {
          setSuggestions([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingSuggestions(false);
        }
      }

      try {
        const payload = await getHealth();
        if (!cancelled) {
          setHealth(payload);
        }
      } catch {
        if (!cancelled) {
          setHealth(null);
        }
      } finally {
        if (!cancelled) {
          setCheckingHealth(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!busy) {
      return;
    }
    const started = Date.now();
    setElapsed(0);
    const timer = setInterval(() => setElapsed(Date.now() - started), TICK_MS);
    return () => clearInterval(timer);
  }, [busy]);

  const ask = useCallback(
    async (question: string) => {
      if (busy) {
        return;
      }
      const id = nextId.current++;
      setTurns((current) => [
        ...current,
        { id, question, response: null, error: null },
      ]);
      setDraft("");
      setBusy(true);

      try {
        const response = await askQuestion(question);
        setTurns((current) =>
          current.map((turn) =>
            turn.id === id ? { ...turn, response } : turn,
          ),
        );
      } catch (caught) {
        const error =
          caught instanceof ApiError
            ? caught
            : new ApiError("An unexpected error occurred.");
        setTurns((current) =>
          current.map((turn) => (turn.id === id ? { ...turn, error } : turn)),
        );
      } finally {
        setBusy(false);
      }
    },
    [busy],
  );

  // Scroll the newest question to the top rather than scrolling to the input:
  // the input is sticky and always visible, and an answer can be several
  // screens long, so ending at the bottom would hide the headline that is the
  // point of the whole exchange.
  useEffect(() => {
    if (turns.length > 0) {
      latestTurnRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [turns.length]);

  const latest = turns.length > 0 ? turns[turns.length - 1] : null;
  const traceResponse = latest?.response ?? null;
  const showColdStartHint = busy && elapsed >= ANALYSIS_COLD_START_HINT_MS;

  return (
    <div className="min-h-screen">
      <Header health={health} checking={checkingHealth} />

      <main className="page-glow mx-auto max-w-[92rem] px-5 pb-16 pt-8 sm:px-8">
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1.62fr_1fr]">
          {/* Conversation */}
          <div className="min-w-0">
            {turns.length === 0 && (
              <EmptyState hasSuggestions={suggestions.length > 0} />
            )}

            <div className="space-y-8">
              {turns.map((turn, index) => (
                <div
                  key={turn.id}
                  ref={index === turns.length - 1 ? latestTurnRef : undefined}
                  className="scroll-mt-20 space-y-4"
                >
                  <div className="flex justify-end">
                    <p className="max-w-[85%] rounded-xl rounded-br-sm border border-border bg-raised px-4 py-2.5 text-sm leading-relaxed text-ink">
                      {turn.question}
                    </p>
                  </div>

                  {turn.response !== null && (
                    <>
                      <AnswerCard
                        response={turn.response}
                        onRetry={(question) => void ask(question)}
                      />
                      {/* On narrow screens the trace lives with its answer. */}
                      <details className="rounded-xl border border-border bg-surface lg:hidden">
                        <summary className="cursor-pointer px-5 py-3 text-xs font-semibold uppercase tracking-[0.14em] text-faint focus:outline-none focus-visible:ring-2 focus-visible:ring-accent">
                          Agent trace · {turn.response.trace.steps.length} steps
                        </summary>
                        <div className="border-t border-border p-3">
                          <TracePanel
                            response={turn.response}
                            busy={false}
                            elapsedMs={0}
                            hideHeader
                          />
                        </div>
                      </details>
                    </>
                  )}

                  {turn.error !== null && (
                    <ErrorCard
                      error={turn.error}
                      question={turn.question}
                      onRetry={(question) => void ask(question)}
                    />
                  )}

                  {turn.response === null && turn.error === null && busy && (
                    <div className="rounded-xl border border-border bg-surface p-6">
                      <div className="flex items-center gap-3">
                        <span className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-accent" />
                        <p className="text-sm text-muted">
                          Running the agent pipeline…
                          <span className="ml-2 font-mono tabular-nums text-faint">
                            {(elapsed / 1000).toFixed(1)}s
                          </span>
                        </p>
                      </div>
                      {showColdStartHint && (
                        <p className="mt-3 max-w-lg text-xs leading-relaxed text-faint">
                          This is taking longer than usual. The backend runs on
                          a free tier that suspends idle services, so the first
                          request of the day can spend up to 50 seconds waking
                          it. Nothing has failed yet.
                        </p>
                      )}
                      <div className="mt-4 lg:hidden">
                        <TracePanel
                          response={null}
                          busy
                          elapsedMs={elapsed}
                          hideHeader
                        />
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="sticky bottom-0 -mx-1 pt-8">
              <div className="bg-base/85 px-1 pb-4 pt-2 backdrop-blur">
                <QuestionInput
                  value={draft}
                  onChange={setDraft}
                  onSubmit={(question) => void ask(question)}
                  busy={busy}
                />
                <div className="mt-4">
                  <SuggestionChips
                    questions={suggestions}
                    onSelect={(question) => void ask(question)}
                    busy={busy}
                    loading={loadingSuggestions}
                    compact={turns.length > 0}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Trace panel.
              min-w-0 is load-bearing, not tidiness: a grid item defaults to a
              min-width of min-content, and the SQL blocks contain long
              unbreakable lines. Without it the trace column claims the width of
              its widest query and the conversation column collapses to zero. */}
          <aside className="hidden min-w-0 lg:block">
            <div className="sticky top-20">
              <TracePanel
                response={traceResponse}
                busy={busy}
                elapsedMs={elapsed}
              />
            </div>
          </aside>
        </div>

        <footer className="mt-16 border-t border-border pt-6">
          <p className="max-w-3xl text-xs leading-relaxed text-faint">
            Data covers 1 August 2025 to 31 July 2026 across 50 stores, 20,000
            orders and 8 cities. &ldquo;Today&rdquo; is fixed at 31 July 2026,
            so relative periods resolve against the dataset rather than the
            calendar. All revenue figures are net of the 5% tax and shown in
            INR.
          </p>
        </footer>
      </main>
    </div>
  );
}
