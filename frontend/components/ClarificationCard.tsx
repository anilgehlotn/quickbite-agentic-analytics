"use client";

/**
 * A question put back to the user, with one-click ways to answer it.
 *
 * This is deliberately not styled as an error. The system asking which of two
 * readings was meant is it working correctly on an underspecified question,
 * and colouring that like a failure would teach a reader that asking loosely
 * breaks things — when the whole point of a natural-language interface is that
 * they can.
 *
 * The interpretations are complete questions, so clicking one submits it
 * verbatim as the next turn. A reader can see exactly what they will get
 * before choosing, which is the difference between a clarification and a
 * guessing game.
 */

import type { Clarification } from "@/lib/types";

/**
 * Render a clarifying question and its interpretations.
 *
 * @param props.clarification - What to ask and what to offer.
 * @param props.onSelect - Called with the chosen question, submitted as-is.
 * @param props.busy - Whether a request is in flight; disables the options.
 * @returns The rendered card.
 */
export default function ClarificationCard({
  clarification,
  onSelect,
  busy,
}: {
  clarification: Clarification;
  onSelect: (question: string) => void;
  busy: boolean;
}): JSX.Element {
  return (
    <article className="rounded-lg border border-border bg-surface p-6 sm:p-8">
      <div className="flex items-center gap-2">
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-accent" />
        <span className="label-caps text-accent">Needs one detail</span>
      </div>

      <h3 className="mt-3 text-balance text-lg font-semibold leading-snug tracking-tight text-ink sm:text-xl">
        {clarification.question}
      </h3>

      <p className="mt-3 text-sm leading-relaxed text-muted">
        {clarification.reason}
      </p>

      <div className="mt-6">
        <h4 className="label-caps">Ask instead</h4>
        <ul className="mt-3 space-y-2">
          {clarification.options.map((option) => (
            <li key={option}>
              <button
                type="button"
                disabled={busy}
                onClick={() => onSelect(option)}
                aria-label={`Ask: ${option}`}
                className="group flex w-full items-start gap-3 rounded border border-border bg-canvas px-4 py-3 text-left text-sm leading-relaxed text-muted transition-colors hover:border-accent-line hover:bg-accent-soft hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface disabled:cursor-not-allowed disabled:opacity-50"
              >
                <span
                  aria-hidden="true"
                  className="mt-[0.4375rem] h-1 w-1 shrink-0 rounded-full bg-accent"
                />
                <span>{option}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </article>
  );
}
