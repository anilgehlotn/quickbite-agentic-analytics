"use client";

/**
 * The eight evaluation questions, as one-click chips.
 *
 * These are the most important control on the page. A reviewer's first
 * interaction should be a single click that always succeeds, not a blank box
 * they have to invent a question for — and because these eight are warmed in
 * the answer cache, they return instantly and correctly even when no model
 * provider is reachable.
 *
 * The cached ones are marked, which is honest rather than decorative: it tells
 * the reader why the answer appeared immediately.
 */

import type { QuestionSuggestion } from "@/lib/types";

/**
 * Clickable chips for the canonical questions.
 *
 * @param props.questions - The suggestions from the API.
 * @param props.onSelect - Called with the full question text on click.
 * @param props.busy - Whether a request is in flight; disables the chips.
 * @param props.loading - Whether the suggestions are still being fetched.
 * @returns The rendered chips, or a placeholder row while loading.
 */
export default function SuggestionChips({
  questions,
  onSelect,
  busy,
  loading,
  compact = false,
}: {
  questions: QuestionSuggestion[];
  onSelect: (question: string) => void;
  busy: boolean;
  loading: boolean;
  compact?: boolean;
}): JSX.Element | null {
  if (loading) {
    return (
      <div className="flex flex-wrap gap-2" aria-hidden="true">
        {Array.from({ length: 6 }).map((_, index) => (
          <span
            key={index}
            className="h-8 w-40 animate-pulse-soft rounded-full border border-border bg-surface"
          />
        ))}
      </div>
    );
  }

  if (questions.length === 0) {
    return null;
  }

  // Once the conversation has started the chips become a single scrollable
  // line. They stay reachable, but they stop occupying a third of the sticky
  // footer and crowding out the answer above it.
  return (
    <div>
      {!compact && (
        <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-faint">
          Try one of these
        </h2>
      )}
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
              className="group inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3.5 py-2 text-sm text-muted transition-colors hover:border-accent/40 hover:bg-raised hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-50"
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
