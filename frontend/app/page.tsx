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
import ClarificationCard from "@/components/ClarificationCard";
import Header from "@/components/Header";
import ModeNotice from "@/components/ModeNotice";
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
  HEALTH_PROBE_TIMEOUT_MS,
  sleep,
  WAKE_DEADLINE_MS,
  WAKING_AFTER_MS,
  wakeRetryDelay,
} from "@/lib/api";
import type { ConnectionPhase } from "@/lib/api";
import { CANONICAL_QUESTIONS } from "@/lib/questions";
import type {
  AnalysisResponse,
  HealthResponse,
  QuestionSuggestion,
} from "@/lib/types";

/** How often the in-flight elapsed timer ticks, in milliseconds. */
const TICK_MS = 250;

/**
 * The suggestions as they are known before the backend answers.
 *
 * `cached` is false rather than true: every one of these is in fact warmed in
 * the committed cache, but that is a property of the deployment this bundle is
 * talking to, and claiming it unverified would be exactly the kind of
 * confidently-wrong statement the rest of the system exists to avoid. The
 * marker appears a moment later, when /api/questions confirms it.
 */
const INITIAL_SUGGESTIONS: QuestionSuggestion[] = CANONICAL_QUESTIONS.map(
  (entry) => ({ ...entry, cached: false }),
);

/**
 * Ask a question, holding the request open while the backend is still waking.
 *
 * The input is never disabled by the connection state, which means a question
 * can be asked before the backend is listening at all. When that happens the
 * browser fails the fetch immediately — a refused connection is not a slow
 * one — and showing that as "the request did not complete" would be wrong
 * twice over: nothing has failed, and the answer is seconds away.
 *
 * So a transport failure is retried on the same schedule and the same budget
 * the health probe uses. The turn stays in its running state throughout, which
 * is what queueing looks like from the reader's side.
 *
 * Only failures that never reached the server are retried. A 4xx, a 5xx or a
 * rate limit means the backend answered and the answer was no; a timeout means
 * it was already given a full minute. Those surface at once.
 *
 * @param question - The question to ask.
 * @returns The analysis, once the backend produces one.
 * @throws ApiError when the server answered with a failure, or when the wake
 *   budget is exhausted without the server ever answering.
 */
async function askWhileWaking(question: string): Promise<AnalysisResponse> {
  const deadline = Date.now() + WAKE_DEADLINE_MS;
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await askQuestion(question);
    } catch (caught) {
      const neverReachedServer =
        caught instanceof ApiError &&
        caught.status === undefined &&
        !caught.timedOut;
      if (!neverReachedServer || Date.now() >= deadline) {
        throw caught;
      }
      await sleep(wakeRetryDelay(attempt));
    }
  }
}

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
      className={`rounded-lg border p-6 ${
        rateLimited
          ? "border-caution-line bg-caution-soft"
          : "border-negative-line bg-negative-soft"
      }`}
      role="alert"
    >
      <h3
        className={`text-sm font-semibold ${
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
          className="rounded border border-accent-line bg-surface px-3.5 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent hover:text-accent-fg focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-canvas"
        >
          Try again
        </button>
        {error.requestId !== undefined && (
          <span className="font-mono text-label text-faint">
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
 * This is the screen a reviewer judges in the first five seconds, so it is the
 * one place the page states its case at full size rather than in a card. It
 * spans the full width above the two-column grid: a centred headline, one line
 * describing what actually happens when you ask, and nothing else competing
 * with them.
 *
 * @param props.waking - Whether the backend is still being woken, which adds
 *   the explanation of why the first request may be slow.
 * @returns The rendered introduction.
 */
function EmptyState({ waking }: { waking: boolean }): JSX.Element {
  return (
    <section className="mx-auto max-w-3xl px-1 pb-10 pt-10 text-center sm:pb-14 sm:pt-16">
      <h1 className="text-balance text-[2rem] font-semibold leading-[1.1] tracking-[-0.028em] text-ink sm:text-display">
        Ask anything about QuickBite&rsquo;s performance
      </h1>
      <p className="mx-auto mt-6 max-w-2xl text-pretty text-lede text-muted">
        Four agents answer in sequence: a planner works out what to measure, a
        SQL analyst writes and runs the queries, a verifier checks the numbers
        arithmetically before you see them, and an insight agent explains what
        the result means.
      </p>
      <p className="mt-4 text-sm text-faint">
        Every step, including the exact SQL, is shown in the trace. Pick a
        question below to start.
      </p>

      {/* The waking notice, in a slot whose height is reserved whether or not
          it has anything in it. Reserving it is the point: this line appears a
          few seconds after load and disappears again when the backend answers,
          and without a fixed box the eight cards below would jump twice while
          someone was reading them.
          The two heights are measured, not guessed: the sentence wraps to
          three lines at 380px (59px) and two at the `sm` breakpoint (39px),
          where the paragraph is capped at max-w-xl rather than by the
          viewport. Both reserves carry roughly a line of slack on top of that,
          so a fallback font with wider metrics still fits rather than
          reintroducing the shift this box exists to prevent. */}
      <div
        aria-live="polite"
        className="mx-auto mt-4 flex min-h-[4.25rem] max-w-xl items-start justify-center sm:min-h-[3.25rem]"
      >
        {waking && (
          <p className="text-xs leading-relaxed text-caution">
            Waking the server. The free tier suspends it when idle, so the
            first request can take up to a minute — you can ask now.
          </p>
        )}
      </div>
    </section>
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
  const [suggestions, setSuggestions] =
    useState<QuestionSuggestion[]>(INITIAL_SUGGESTIONS);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [phase, setPhase] = useState<ConnectionPhase>("checking");

  const nextId = useRef(0);
  const latestTurnRef = useRef<HTMLDivElement>(null);

  // Wake the backend on first paint and keep probing until it answers.
  //
  // Nothing on the page waits for this. The cards are already rendered from a
  // build-time constant and are already clickable; the probe exists so that a
  // suspended service starts booting the moment the page opens rather than
  // when someone finally clicks, and so the header can say which of those two
  // things is happening.
  useEffect(() => {
    let cancelled = false;

    // "connecting" is only honest for a moment. After that, silence from a
    // free-tier host means it is booting, and saying so is better than a
    // spinner that could equally mean broken.
    const wakingTimer = setTimeout(() => {
      if (!cancelled) {
        setPhase((current) => (current === "checking" ? "waking" : current));
      }
    }, WAKING_AFTER_MS);

    void (async () => {
      const deadline = Date.now() + WAKE_DEADLINE_MS;
      for (let attempt = 0; !cancelled; attempt += 1) {
        try {
          const payload = await getHealth(HEALTH_PROBE_TIMEOUT_MS);
          if (!cancelled) {
            setHealth(payload);
            setPhase("connected");
          }
          return;
        } catch {
          // A failed probe during a cold start is the expected case, not an
          // error: the service is not listening yet. Only the deadline
          // distinguishes a boot from an outage.
          if (Date.now() >= deadline) {
            if (!cancelled) {
              setHealth(null);
              setPhase("offline");
            }
            return;
          }
          await sleep(wakeRetryDelay(attempt));
        }
      }
    })();

    return () => {
      cancelled = true;
      clearTimeout(wakingTimer);
    };
  }, []);

  // Reconcile the suggestion cards with the backend, in the background.
  //
  // The questions themselves are already on screen. This call adds the one
  // thing the bundle cannot know — which are cached — and corrects the list in
  // the unlikely event that a deployed backend disagrees with the build. A
  // failure here changes nothing: the constant stands.
  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const payload = await getCanonicalQuestions();
        if (!cancelled && payload.questions.length > 0) {
          setSuggestions(payload.questions);
        }
      } catch {
        // Keep the build-time list. The header reports the connection state,
        // so a reader is not left to infer it from the cards.
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
        const response = await askWhileWaking(question);
        setTurns((current) =>
          current.map((turn) =>
            turn.id === id ? { ...turn, response } : turn,
          ),
        );
        // An answer proves the backend is up, which the probe loop may not
        // have noticed yet — it can be mid-backoff. Without this the header
        // would go on saying "waking the server" above a finished answer.
        if (phase !== "connected") {
          setPhase("connected");
          void getHealth(HEALTH_PROBE_TIMEOUT_MS)
            .then(setHealth)
            .catch(() => undefined);
        }
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
    [busy, phase],
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
  const waking = phase === "waking";
  // The in-flight card follows "has the backend answered anything yet", not
  // just the waking label: a request retrying against a server that has not
  // responded is waiting for the server whether the probe currently calls that
  // waking, checking or offline.
  const serverPending = phase !== "connected";
  // While the backend is known not to have answered there is no need to wait
  // fifteen seconds to explain the delay: the reason is already established.
  const showColdStartHint =
    busy && (serverPending || elapsed >= ANALYSIS_COLD_START_HINT_MS);

  return (
    <div className="min-h-screen">
      <Header health={health} phase={phase} />

      <main className="mx-auto max-w-[92rem] px-5 pb-20 sm:px-8">
        {turns.length === 0 && <EmptyState waking={waking} />}

        <div
          className={`grid grid-cols-1 gap-10 lg:grid-cols-[1.62fr_1fr] ${
            turns.length === 0 ? "" : "pt-8"
          }`}
        >
          {/* Conversation */}
          <div className="min-w-0">
            <div className="space-y-10">
              {turns.map((turn, index) => (
                <div
                  key={turn.id}
                  ref={index === turns.length - 1 ? latestTurnRef : undefined}
                  className="scroll-mt-24 space-y-4"
                >
                  <div className="flex justify-end">
                    <p className="max-w-[85%] rounded border border-border bg-raised px-4 py-2.5 text-sm leading-relaxed text-ink">
                      {turn.question}
                    </p>
                  </div>

                  {turn.response !== null && (
                    <>
                      {turn.response.status === "clarification_needed" &&
                      turn.response.clarification !== null ? (
                        <ClarificationCard
                          clarification={turn.response.clarification}
                          onSelect={(question) => void ask(question)}
                          busy={busy}
                        />
                      ) : (
                        <AnswerCard
                          response={turn.response}
                          onRetry={(question) => void ask(question)}
                          onAsk={(question) => void ask(question)}
                          busy={busy}
                        />
                      )}
                      {/* On narrow screens the trace lives with its answer. */}
                      <details className="min-w-0 rounded-lg border border-border bg-surface lg:hidden">
                        <summary className="label-caps cursor-pointer px-5 py-3.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent">
                          Agent trace · {turn.response.trace.steps.length} steps
                        </summary>
                        <div className="min-w-0 border-t border-border p-4">
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
                    <div className="rounded-lg border border-border bg-surface p-6">
                      <div className="flex items-center gap-3">
                        <span
                          aria-hidden="true"
                          className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-accent"
                        />
                        <p className="text-sm text-muted">
                          {serverPending
                            ? "Waiting for the server, then running the agent pipeline…"
                            : "Running the agent pipeline…"}
                          <span className="ml-2 tabular-nums text-faint">
                            {(elapsed / 1000).toFixed(1)}s
                          </span>
                        </p>
                      </div>
                      {showColdStartHint && (
                        <p className="mt-3 max-w-lg text-xs leading-relaxed text-faint">
                          {serverPending
                            ? "The question is queued and will run as soon as the backend answers. It is on a free tier that suspends idle services, so waking it takes up to 50 seconds. Nothing has failed."
                            : "This is taking longer than usual. The backend runs on a free tier that suspends idle services, so the first request of the day can spend up to 50 seconds waking it. Nothing has failed yet."}
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

            {/* The composer pins to the bottom only once there is a
                conversation to scroll behind it. In the empty state the cards
                are the tallest thing on the page, and pinning a block taller
                than the viewport makes it lurch rather than stick. */}
            <div
              className={
                turns.length === 0 ? "-mx-1 pt-2" : "sticky bottom-0 -mx-1 pt-2"
              }
            >
              {/* A short fade so an answer scrolls out under the composer
                  instead of being sliced off by a hard edge. */}
              {turns.length > 0 && (
                <div
                  aria-hidden="true"
                  className="pointer-events-none h-8 bg-gradient-to-b from-transparent to-canvas"
                />
              )}
              <div className="bg-canvas px-1 pb-5 pt-2">
                {/* Only once the connection has actually resolved. Showing the
                    offline banner while the backend is merely booting would
                    report an outage that is not happening, which is the exact
                    failure this page is meant to avoid. */}
                {(phase === "connected" || phase === "offline") && (
                  <ModeNotice
                    mode={health?.mode ?? "offline"}
                    cachedAnswers={health?.cached_answers ?? 0}
                  />
                )}
                <QuestionInput
                  value={draft}
                  onChange={setDraft}
                  onSubmit={(question) => void ask(question)}
                  busy={busy}
                />
                <div className={turns.length === 0 ? "mt-8" : "mt-4"}>
                  <SuggestionChips
                    questions={suggestions}
                    onSelect={(question) => void ask(question)}
                    busy={busy}
                    compact={turns.length > 0 && health?.mode === "full"}
                    // Cards only before the first question, where they are the
                    // primary call to action. Afterwards they collapse back to
                    // chips so the answers own the space.
                    cards={turns.length === 0}
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

        <footer className="mt-20 border-t border-border pt-6">
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
