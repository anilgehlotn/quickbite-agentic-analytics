/**
 * The eight canonical questions, known at build time.
 *
 * These are fixed: they are the evaluation questions, they are warmed in the
 * committed answer cache, and they change only when the backend's
 * ``CANONICAL_QUESTIONS`` changes. Nothing about them needs to be discovered at
 * runtime, so the page must not wait for a network round trip to show them.
 *
 * That wait was the problem this constant exists to solve. The backend runs on
 * a free tier that suspends when idle, and a suspended service takes up to
 * fifty seconds to wake. Rendering the suggestion cards from ``GET
 * /api/questions`` meant the first visitor after a quiet period saw eight empty
 * skeletons and a "connecting" pill for most of a minute — a page that looks
 * broken at precisely the moment it is being judged.
 *
 * The endpoint is still called, in the background, and the cards reconcile if
 * it disagrees. It carries one thing this file cannot know: which questions are
 * currently cached. That is a live property of the deployment, so the cached
 * marker appears when the answer arrives rather than being asserted here.
 *
 * Because this list is a hand-maintained mirror, it can drift. It cannot drift
 * silently: ``backend/tests/test_frontend_contract.py`` parses this file and
 * fails if the ids, questions or labels differ from the backend's.
 */

/**
 * One canonical question as it is known before the backend answers.
 *
 * Deliberately not {@link QuestionSuggestion}: that type carries `cached`, and
 * at build time the cache state is unknown.
 */
export interface CanonicalQuestion {
  /** Stable identifier, matching the backend's and the golden answers'. */
  id: string;
  /** The question text, sent verbatim when the card is clicked. */
  question: string;
  /** Short heading for the card. */
  label: string;
}

/**
 * The eight evaluation questions, mirroring the backend's CANONICAL_QUESTIONS.
 *
 * Kept in the same order as the backend, because that order is the one the
 * golden answers and the demo script use.
 */
export const CANONICAL_QUESTIONS: readonly CanonicalQuestion[] = [
  {
    id: "q1",
    question: "What was our total revenue, order count and AOV in the last 3 months?",
    label: "Revenue, orders and AOV",
  },
  {
    id: "q2",
    question: "Which stores are performing best and worst by revenue in the last 3 months?",
    label: "Best and worst stores",
  },
  {
    id: "q3",
    question: "How do our sales channels compare in the last 3 months?",
    label: "Channel comparison",
  },
  {
    id: "q4",
    question: "Which products sell the most in the last 3 months?",
    label: "Top products",
  },
  {
    id: "q5",
    question: "Are any cities showing declining revenue in the last 3 months?",
    label: "Declining cities",
  },
  {
    id: "q6",
    question: "How does weekend trading compare with weekday trading?",
    label: "Weekend vs weekday",
  },
  {
    id: "q7",
    question: "How much do festive periods lift trading versus normal days?",
    label: "Festive lift",
  },
  {
    id: "q8",
    question: "Which stores have consistently declining revenue, and why are they declining?",
    label: "Consistent decliners and why",
  },
] as const;
