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
 * Delay after which the UI should explain that the backend may be waking up.
 * Chosen to be longer than a warm response but well short of the timeout.
 */
export const COLD_START_HINT_MS = 4_000;

/**
 * Delay after which an in-flight analysis should mention a sleeping server.
 * A warm analysis takes roughly 20 seconds, so this fires only when the wait
 * has become unusual.
 */
export const ANALYSIS_COLD_START_HINT_MS = 15_000;

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
 * @returns The parsed response body.
 * @throws ApiError when the request times out, the network fails, or the
 *   server returns a non-2xx status.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

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
        `No response within ${REQUEST_TIMEOUT_MS / 1000} seconds. The backend ` +
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
 * @returns The health payload.
 * @throws ApiError when the backend cannot be reached.
 */
export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
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
