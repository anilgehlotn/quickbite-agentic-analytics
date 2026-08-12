"use client";

/**
 * The eight evaluation questions, in one of three presentations.
 *
 * These are the most important control on the page. A reviewer's first
 * interaction should be a single click that always succeeds, not a blank box
 * they have to invent a question for — and because these eight are warmed in
 * the answer cache, they return instantly and correctly even when no model
 * provider is reachable.
 *
 * The three presentations are purely visual and cost nothing in behaviour:
 *
 *   * `cards` — the empty state. Each question gets a real card with its label
 *     as a heading and the full question beneath, because before the first
 *     click the questions *are* the interface and a row of truncated chips
 *     under-sells what the system can do.
 *   * default — a wrapped row of chips, used once a conversation has started
 *     but live analysis is unavailable, so the answerable questions stay
 *     prominent rather than scrolling out of sight.
 *   * `compact` — a single scrollable line, once the conversation is running
 *     normally and the answers above deserve the space.
 *
 * Cached questions are marked, which is honest rather than decorative: it tells
 * the reader why the answer appeared immediately. The marker is a word in the
 * card presentation and a dot with a title where space forbids it, so the
 * status is never carried by colour alone. It appears only once the backend has
 * confirmed the cache state; the questions themselves come from a build-time
 * constant and are on screen before any request is made, so there is no loading
 * state here to render.
 */

import type { QuestionSuggestion } from "@/lib/types";

/** Focus ring shared by both presentations, offset against the page canvas. */
const FOCUS_RING =
  "focus:outline-none focus-visible:ring-2 focus-visible:ring-accent " +
  "focus-visible:ring-offset-2 focus-visible:ring-offset-canvas";

/**
 * Clickable prompts for the canonical questions.
 *
 * @param props.questions - The suggestions to render, seeded from the
 *   build-time constant and reconciled with the API when it answers.
 * @param props.onSelect - Called with the full question text on click.
 * @param props.busy - Whether a request is in flight; disables the controls.
 * @param props.compact - Render as a single scrollable line of chips.
 * @param props.cards - Render as a grid of cards, for the empty state.
 * @returns The rendered suggestions.
 */
export default function SuggestionChips({
  questions,
  onSelect,
  busy,
  compact = false,
  cards = false,
}: {
  questions: QuestionSuggestion[];
  onSelect: (question: string) => void;
  busy: boolean;
  compact?: boolean;
  cards?: boolean;
}): JSX.Element | null {
  if (questions.length === 0) {
    return null;
  }

  if (cards) {
    return (
      <div>
        <h2 className="label-caps">Start with one of these</h2>
        <ul className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          {questions.map((suggestion) => (
            <li key={suggestion.id}>
              <button
                type="button"
                disabled={busy}
                onClick={() => onSelect(suggestion.question)}
                aria-label={`Ask: ${suggestion.question}`}
                className={`group flex h-full w-full flex-col items-start gap-1.5 rounded-lg border border-border bg-surface px-4 py-3.5 text-left transition-colors hover:border-accent-line hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:border-border disabled:hover:bg-surface ${FOCUS_RING}`}
              >
                <span className="flex w-full items-center gap-2">
                  <span className="text-sm font-semibold tracking-tight text-ink">
                    {suggestion.label}
                  </span>
                  {suggestion.cached && (
                    <span
                      className="ml-auto shrink-0 text-micro uppercase tracking-[0.1em] text-positive"
                      title="Answered from cache — instant, and available even if no model provider is reachable"
                    >
                      cached
                    </span>
                  )}
                </span>
                <span className="text-xs leading-relaxed text-muted">
                  {suggestion.question}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  return (
    <div>
      {!compact && <h2 className="label-caps">Try one of these</h2>}
      <ul
        className={
          compact
            ? "-mx-1 flex gap-2 overflow-x-auto px-1 pb-1 [mask-image:linear-gradient(to_right,black_calc(100%-2rem),transparent)] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            : "mt-3 flex flex-wrap gap-2"
        }
      >
        {questions.map((suggestion) => (
          <li key={suggestion.id} className={compact ? "shrink-0" : undefined}>
            <button
              type="button"
              disabled={busy}
              onClick={() => onSelect(suggestion.question)}
              title={suggestion.question}
              aria-label={`Ask: ${suggestion.question}`}
              className={`group inline-flex items-center gap-2 rounded border border-border bg-surface px-3 py-1.5 text-xs text-muted transition-colors hover:border-accent-line hover:bg-accent-soft hover:text-accent disabled:cursor-not-allowed disabled:opacity-50 ${FOCUS_RING}`}
            >
              {suggestion.cached && (
                <span
                  className="h-1.5 w-1.5 shrink-0 rounded-full bg-positive"
                  title="Answered from cache — instant, and available even if no model provider is reachable"
                  aria-hidden="true"
                />
              )}
              {suggestion.label}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
