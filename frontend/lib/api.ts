/**
 * Typed client for the QuickBite analytics API.
 *
 * The base URL comes from NEXT_PUBLIC_API_URL so the same build can point at a
 * local backend or the deployed one without a code change. Vercel injects it at
 * build time; locally it comes from .env.local.
 *
 * Every failure path produces an {@link ApiError} carrying the backend's own
 * message and, where the backend supplied one, its request id. That id appears
 * in the server logs verbatim, so a user reporting a problem can quote
 * something that locates it exactly.
 */

import type {
  AnalysisResponse,
  DataQualityResponse,
  ErrorPayload,
  HealthResponse,
  QuestionsResponse,
  SchemaResponse,
} from "@/lib/types";

/** Base URL of the backend API, without a trailing slash. */
export const API_URL: string = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

/**
 * Request timeout in milliseconds.
 *
 * Render's free tier suspends idle services and a cold start can take close to
 * 50 seconds. The timeout sits above that so a waking backend resolves rather
 * than aborting at the worst possible moment and reporting a false outage.
 */
export const REQUEST_TIMEOUT_MS = 60_000;

/**
 * Delay after which an in-flight analysis should mention a sleeping server.
 * A warm analysis takes roughly 20 seconds, so this fires only when the wait
 * has become unusual.
 */
export const ANALYSIS_COLD_START_HINT_MS = 15_000;

/**
 * Per-attempt timeout for a health probe during the wake sequence.
 *
 * Much shorter than {@link REQUEST_TIMEOUT_MS} on purpose. A single 60-second
 * attempt against a suspended service tells us nothing for a minute; several
 * short attempts let the caller retry, report progress, and notice the moment
 * the service starts answering.
 */
export const HEALTH_PROBE_TIMEOUT_MS = 10_000;

/**
 * How long a pending health check may run before the UI stops saying
 * "connecting" and starts saying the server is waking.
 *
 * A warm backend answers in well under a second, so three seconds of silence
 * means something slower is happening and the reader deserves to be told what.
 */
export const WAKING_AFTER_MS = 3_000;

/**
 * How long to keep retrying the health probe before reporting an outage.
 *
 * A cold start costs up to fifty seconds. Ninety gives that comfortable room:
 * declaring the backend offline while it is merely booting would be the single
 * most misleading thing this page could say.
 */
export const WAKE_DEADLINE_MS = 90_000;

/** Delay before the second health attempt, in milliseconds. */
export const WAKE_RETRY_BASE_MS = 1_500;

/** Ceiling on the backoff between health attempts, in milliseconds. */
export const WAKE_RETRY_MAX_MS = 8_000;

/**
 * Delay before health attempt number `attempt`, with exponential backoff.
 *
 * Starts quick so a backend that is nearly up is noticed immediately, then
 * eases off so ninety seconds of waiting is not ninety seconds of requests.
 *
 * @param attempt - Zero-based index of the attempt that just failed.
 * @returns Milliseconds to wait before the next attempt.
 */
export function wakeRetryDelay(attempt: number): number {
  return Math.min(WAKE_RETRY_BASE_MS * 2 ** attempt, WAKE_RETRY_MAX_MS);
}

/**
 * Resolve after a delay.
 *
 * @param ms - Milliseconds to wait.
 * @returns A promise that resolves when the delay elapses.
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * How far the page has got in reaching the backend.
 *
 * Four states rather than a boolean because "not connected yet" and "not
 * reachable" are different facts and only one of them is bad news. A suspended
 * free-tier service reported as an outage is a lie the page told for the fifty
 * seconds it took to wake.
 *
 * * `checking` — a probe is in flight and has been for less than
 *   {@link WAKING_AFTER_MS}. Normal; says nothing yet.
 * * `waking` — probes are still failing or pending. The service is very likely
 *   booting, and the page says so.
 * * `connected` — health answered. What the service can actually do is then a
 *   question for `HealthResponse.mode`, not for this type.
 * * `offline` — {@link WAKE_DEADLINE_MS} elapsed with no answer. Only now is it
 *   honest to call it unreachable.
 */
export type ConnectionPhase = "checking" | "waking" | "connected" | "offline";

/** An API call that failed, carrying a message suitable for display. */
export class ApiError extends Error {
  /** HTTP status when the server responded, otherwise undefined. */
  readonly status?: number;

  /** True when the request exceeded REQUEST_TIMEOUT_MS. */
  readonly timedOut: boolean;

  /** Correlation id from the response body or header, when present. */
  readonly requestId?: string;

  /** Machine-readable error code from the backend, e.g. "rate_limited". */
  readonly code?: string;

  /** Seconds to wait before retrying, from a 429 response. */
  readonly retryAfter?: number;

  /**
   * @param message - Human-readable explanation.
   * @param options - Status, timeout flag and backend correlation details.
   */
  constructor(
    message: string,
    options: {
      status?: number;
      timedOut?: boolean;
      requestId?: string;
      code?: string;
      retryAfter?: number;
    } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.timedOut = options.timedOut ?? false;
    this.requestId = options.requestId;
    this.code = options.code;
    this.retryAfter = options.retryAfter;
  }

  /** True when this failure was a rate limit rather than a fault. */
  get isRateLimited(): boolean {
    return this.status === 429;
  }
}

/**
 * Narrow an unknown parsed body to the API's structured error shape.
 *
 * @param body - The parsed response body.
 * @returns The error payload, or null when the body is not one.
 */
function asErrorPayload(body: unknown): ErrorPayload | null {
  if (typeof body !== "object" || body === null) {
    return null;
  }
  const candidate = body as Partial<ErrorPayload>;
  if (typeof candidate.message !== "string") {
    return null;
  }
  return {
    error: typeof candidate.error === "string" ? candidate.error : "error",
    message: candidate.message,
    request_id:
      typeof candidate.request_id === "string" ? candidate.request_id : "",
    detail: candidate.detail,
  };
}

/**
 * Read a retry delay from a 429 response.
 *
 * @param response - The refused response.
 * @param body - Its parsed body, which may repeat the delay under detail.
 * @returns Seconds to wait, or undefined when not stated.
 */
function readRetryAfter(response: Response, body: unknown): number | undefined {
  const header = response.headers.get("Retry-After");
  if (header !== null && header.trim() !== "") {
    const parsed = Number.parseInt(header, 10);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  if (typeof body === "object" && body !== null) {
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "object" && detail !== null) {
      const value = (detail as { retry_after?: unknown }).retry_after;
      if (typeof value === "number") {
        return value;
      }
    }
  }
  return undefined;
}

/**
 * Fetch JSON from the API with a timeout and typed errors.
 *
 * @param path - Path beginning with a slash, e.g. "/api/health".
 * @param init - Optional method and body for a mutating request.
 * @param timeoutMs - Abort deadline. Defaults to {@link REQUEST_TIMEOUT_MS};
 *   the wake sequence passes a shorter one so it can retry instead of waiting.
 * @returns The parsed response body.
 * @throws ApiError when the request times out, the network fails, or the
 *   server returns a non-2xx status.
 */
async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...(init?.body !== undefined
          ? { "Content-Type": "application/json" }
          : {}),
        ...init?.headers,
      },
      cache: "no-store",
    });

    const headerRequestId = response.headers.get("X-Request-ID") ?? undefined;

    if (!response.ok) {
      let body: unknown = null;
      try {
        body = await response.json();
      } catch {
        body = null;
      }
      const payload = asErrorPayload(body);
      throw new ApiError(
        payload?.message ??
          `The API responded with ${response.status} ${response.statusText}.`,
        {
          status: response.status,
          requestId: payload?.request_id || headerRequestId,
          code: payload?.error,
          retryAfter: readRetryAfter(response, body),
        },
      );
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(
        `No response within ${timeoutMs / 1000} seconds. The backend ` +
          `may be asleep or unreachable.`,
        { timedOut: true },
      );
    }
    throw new ApiError(
      `Could not reach the API at ${API_URL}. Check that the backend is ` +
        `running and that this origin is allowed by its CORS configuration.`,
    );
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Ask the agent pipeline a business question.
 *
 * A run that produces no answer is not an error: it returns an
 * {@link AnalysisResponse} with `answered: false`, an explanation and a trace.
 * Only transport failures, validation rejections and rate limits throw.
 *
 * @param question - The question in natural language.
 * @param useCache - Whether a previously computed answer may be served.
 * @returns The full analysis, including the agent trace.
 * @throws ApiError on a network failure, a 422, or a 429.
 */
export async function askQuestion(
  question: string,
  useCache = true,
): Promise<AnalysisResponse> {
  return request<AnalysisResponse>("/api/ask", {
    method: "POST",
    body: JSON.stringify({ question, use_cache: useCache }),
  });
}

/**
 * Fetch the eight canonical evaluation questions.
 *
 * @returns The suggestions, each flagged with whether it is already cached.
 * @throws ApiError when the backend cannot be reached.
 */
export async function getCanonicalQuestions(): Promise<QuestionsResponse> {
  return request<QuestionsResponse>("/api/questions");
}

/**
 * Fetch backend health, including a live row count from the database.
 *
 * @param timeoutMs - Abort deadline for this probe. The wake sequence passes
 *   {@link HEALTH_PROBE_TIMEOUT_MS} so a suspended backend fails fast enough to
 *   be retried rather than occupying the full request timeout.
 * @returns The health payload.
 * @throws ApiError when the backend cannot be reached.
 */
export async function getHealth(
  timeoutMs: number = REQUEST_TIMEOUT_MS,
): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", undefined, timeoutMs);
}

/**
 * Fetch the data quality gate's report.
 *
 * @returns Every check the gate ran, with its verdict.
 * @throws ApiError when the backend cannot be reached.
 */
export async function getDataQuality(): Promise<DataQualityResponse> {
  return request<DataQualityResponse>("/api/verify");
}

/**
 * Fetch the semantic layer's compact schema.
 *
 * @returns The schema payload.
 * @throws ApiError when the backend cannot be reached.
 */
export async function getSchema(): Promise<SchemaResponse> {
  return request<SchemaResponse>("/api/schema");
}
