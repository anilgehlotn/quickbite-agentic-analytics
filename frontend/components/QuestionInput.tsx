"use client";

/**
 * The question box.
 *
 * Enter submits and Shift+Enter inserts a newline, which is the convention
 * every chat interface has trained people to expect. The textarea grows with
 * its content up to a cap so a long question stays fully visible while typing.
 */

import { useCallback, useEffect, useRef, useState } from "react";

/** Maximum question length, matching the backend's validation. */
export const MAX_QUESTION_LENGTH = 500;

/** Minimum question length, matching the backend's validation. */
export const MIN_QUESTION_LENGTH = 3;

/** Characters remaining at which the counter appears. */
const COUNTER_THRESHOLD = 80;

/** Height cap for the auto-growing textarea, in pixels. */
const MAX_TEXTAREA_HEIGHT = 180;

/**
 * A controlled question input with a send button.
 *
 * @param props.onSubmit - Called with the trimmed question when submitted.
 * @param props.busy - Whether a request is in flight; disables the control.
 * @param props.value - The current text.
 * @param props.onChange - Called when the text changes.
 * @returns The rendered input.
 */
export default function QuestionInput({
  onSubmit,
  busy,
  value,
  onChange,
}: {
  onSubmit: (question: string) => void;
  busy: boolean;
  value: string;
  onChange: (value: string) => void;
}): JSX.Element {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [focused, setFocused] = useState(false);

  const trimmed = value.trim();
  const tooShort = trimmed.length > 0 && trimmed.length < MIN_QUESTION_LENGTH;
  const canSubmit = !busy && trimmed.length >= MIN_QUESTION_LENGTH;
  const remaining = MAX_QUESTION_LENGTH - value.length;
  const showCounter = remaining <= COUNTER_THRESHOLD;

  useEffect(() => {
    const element = textareaRef.current;
    if (element === null) {
      return;
    }
    element.style.height = "auto";
    element.style.height = `${Math.min(
      element.scrollHeight,
      MAX_TEXTAREA_HEIGHT,
    )}px`;
  }, [value]);

  const submit = useCallback(() => {
    if (canSubmit) {
      onSubmit(trimmed);
    }
  }, [canSubmit, onSubmit, trimmed]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submit();
      }
    },
    [submit],
  );

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      className={`rounded-lg border bg-surface shadow-input transition-colors ${
        focused ? "border-accent" : "border-border-strong"
      }`}
    >
      <label htmlFor="question" className="sr-only">
        Ask a question about the business
      </label>
      <div className="flex items-end gap-3 p-3">
        <textarea
          id="question"
          ref={textareaRef}
          rows={1}
          value={value}
          disabled={busy}
          maxLength={MAX_QUESTION_LENGTH}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="Ask about revenue, stores, channels, products, seasonality…"
          aria-describedby="question-hint"
          className="min-h-[2.5rem] flex-1 resize-none bg-transparent px-1 py-2 text-[0.9375rem] leading-relaxed text-ink placeholder:text-faint focus:outline-none disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={!canSubmit}
          aria-label={busy ? "Analysing" : "Send question"}
          // Disabled keeps the accent at low opacity rather than turning grey:
          // the button is the page's only primary action, and a grey rest
          // state means the accent never appears until someone has typed.
          className="mb-0.5 inline-flex h-9 shrink-0 items-center gap-2 rounded bg-accent px-4 text-sm font-medium text-accent-fg transition-colors hover:bg-accent-hover focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface disabled:cursor-not-allowed disabled:opacity-35"
        >
          {busy ? (
            <>
              <span className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-accent-fg" />
              Analysing
            </>
          ) : (
            "Ask"
          )}
        </button>
      </div>

      <div
        id="question-hint"
        className="flex items-center justify-between gap-4 border-t border-border px-4 py-2 text-label text-faint"
      >
        <span>
          {tooShort
            ? `At least ${MIN_QUESTION_LENGTH} characters.`
            : "Enter to send · Shift+Enter for a new line"}
        </span>
        {showCounter && (
          <span
            className={`tabular-nums ${
              remaining <= 0 ? "text-negative" : "text-caution"
            }`}
            aria-live="polite"
          >
            {remaining} left
          </span>
        )}
      </div>
    </form>
  );
}
