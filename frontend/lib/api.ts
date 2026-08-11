/**
 * Typed client for the QuickBite analytics API.
 *
 * The base URL comes from NEXT_PUBLIC_API_URL so the same build can point at a
 * local backend or the deployed one without a code change. Vercel injects it at
 * build time; locally it comes from .env.local.
 */

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

/** Health payload returned by GET /health. */
export interface HealthResponse {
  status: "ok" | "degraded";
  database_ready: boolean;
  fact_orders_rows: number | null;
  data_asof: string;
  providers_configured: string[];
  environment: string;
  version: string;
  database_error?: string;
}

/** Schema payload returned by GET /api/schema. */
export interface SchemaResponse {
  compact_schema: string;
  metrics: string[];
  tables: string[];
  data_asof: string;
  data_start: string;
  revenue_metric: string;
}

/** An API call that failed, carrying a message suitable for display. */
export class ApiError extends Error {
  /** HTTP status when the server responded, otherwise undefined. */
  readonly status?: number;

  /** True when the request exceeded REQUEST_TIMEOUT_MS. */
  readonly timedOut: boolean;

  /**
   * @param message - Human-readable explanation.
   * @param options - Status code and timeout flag.
   */
  constructor(
    message: string,
    options: { status?: number; timedOut?: boolean } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.timedOut = options.timedOut ?? false;
  }
}

/**
 * Fetch JSON from the API with a timeout and typed errors.
 *
 * @param path - Path beginning with a slash, e.g. "/health".
 * @returns The parsed response body.
 * @throws ApiError when the request times out, the network fails, or the
 *   server returns a non-2xx status.
 */
async function request<T>(path: string): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_URL}${path}`, {
      signal: controller.signal,
      headers: { Accept: "application/json" },
      cache: "no-store",
    });

    if (!response.ok) {
      throw new ApiError(
        `The API responded with ${response.status} ${response.statusText}.`,
        { status: response.status },
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
 * Fetch backend health, including a live row count from the database.
 *
 * @returns The health payload.
 * @throws ApiError when the backend cannot be reached.
 */
export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
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
