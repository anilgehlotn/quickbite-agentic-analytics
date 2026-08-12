/**
 * TypeScript mirrors of the backend's Pydantic contracts.
 *
 * These are hand-maintained against `backend/app/agents/contracts.py` and
 * `backend/app/api/routes.py`. A field renamed on one side and not the other is
 * a runtime failure that TypeScript cannot catch for us, because the JSON
 * arriving over the wire is unchecked at the boundary — so the field names and
 * optionality here follow the Python definitions literally, including the
 * `| null` on every field the backend declares as `X | None`.
 *
 * Enum members are string unions rather than TypeScript enums: the wire format
 * is a string, and a union keeps the value and the type identical.
 */

/** What kind of question the user asked. Mirrors QueryIntent. */
export type QueryIntent =
  | "aggregate"
  | "ranking"
  | "comparison"
  | "trend"
  | "diagnostic"
  | "unsupported"
  | "ambiguous";

/**
 * What kind of reply the pipeline produced. Mirrors ResponseStatus.
 *
 * Four states rather than a boolean, because a clarification request and a
 * genuine failure are both "not answered" but mean opposite things to a
 * reader: one is the system working correctly on an underspecified question,
 * the other is an outage.
 */
export type ResponseStatus =
  | "answered"
  | "clarification_needed"
  | "unsupported"
  | "failed";

/** Outcome of verifying a set of query results. Mirrors VerificationStatus. */
export type VerificationStatus = "passed" | "passed_with_warnings" | "failed";

/** The chart shape that best fits a result set. Mirrors ChartType. */
export type ChartType = "bar" | "line" | "grouped_bar" | "none";

/** Execution state of one agent step. Mirrors AgentStatus. */
export type AgentStatus =
  "pending" | "running" | "succeeded" | "failed" | "skipped";

/** How much a failed verification check matters. */
export type CheckSeverity = "error" | "warning" | "info";

/** A single cell in a query result row. */
export type CellValue = string | number | boolean | null;

/** One row of a query result, keyed by column name. */
export type ResultRow = Record<string, CellValue>;

/** The date range a question covers, resolved against the fixed anchor. */
export interface TimeWindow {
  start_date: string;
  end_date: string;
  label: string;
  comparison_start: string | null;
  comparison_end: string | null;
}

/** One piece of analysis within a plan. */
export interface SubQuery {
  id: string;
  purpose: string;
  tables: string[];
  metrics: string[];
  dimensions: string[];
  filters: Record<string, unknown>;
}

/** The planner agent's output: how the question will be answered. */
export interface AnalysisPlan {
  question: string;
  intent: QueryIntent;
  time_window: TimeWindow;
  metrics: string[];
  dimensions: string[];
  sub_queries: SubQuery[];
  requires_diagnostics: boolean;
  reasoning: string;
  confidence: number;
}

/** The outcome of running one sub-query, including the exact SQL. */
export interface QueryResult {
  sub_query_id: string;
  sql: string;
  columns: string[];
  rows: ResultRow[];
  row_count: number;
  execution_ms: number;
  error: string | null;
  attempts: number;
  degraded: boolean;
}

/** One assertion made about the query results. */
export interface VerificationCheck {
  name: string;
  passed: boolean;
  severity: CheckSeverity;
  message: string;
  details: Record<string, unknown> | null;
}

/** The verifier agent's judgement on a set of results. */
export interface VerificationReport {
  status: VerificationStatus;
  checks: VerificationCheck[];
  summary: string;
}

/** The explanation agent's output: what the numbers mean. */
export interface Insight {
  headline: string;
  narrative: string;
  key_findings: string[];
  caveats: string[];
  recommended_actions: string[];
  confidence: number;
}

/** How to visualise a result set. */
export interface ChartSpec {
  chart_type: ChartType;
  x_field: string;
  y_fields: string[];
  title: string;
  series_field: string | null;
}

/** One agent's execution record within a run. */
export interface AgentStep {
  agent_name: string;
  status: AgentStatus;
  started_at: string;
  duration_ms: number;
  summary: string;
  error: string | null;
  llm_provider: string | null;
  tokens: number | null;
}

/** The full record of one run, step by step. */
export interface AgentTrace {
  steps: AgentStep[];
  total_duration_ms: number;
  total_tokens: number;
  providers_used: string[];
}

/** The complete answer to one question: the top-level API response. */
export interface AnalysisResponse {
  question: string;
  answered: boolean;
  plan: AnalysisPlan | null;
  query_results: QueryResult[];
  verification: VerificationReport | null;
  insight: Insight | null;
  chart: ChartSpec | null;
  trace: AgentTrace;
  data_asof: string;
  from_cache: boolean;
  request_id: string | null;
  error: string | null;
  status: ResponseStatus;
  clarification: Clarification | null;
  suggested_questions: string[];
}

/** A question put back to the user, with ways to answer it. */
export interface Clarification {
  question: string;
  reason: string;
  options: string[];
}

/** One suggested question for the frontend's chips. */
export interface QuestionSuggestion {
  id: string;
  question: string;
  label: string;
  cached: boolean;
}

/** The canonical evaluation questions. */
export interface QuestionsResponse {
  questions: QuestionSuggestion[];
  count: number;
}

/** Configuration state of one LLM provider. Keys are never exposed. */
export interface ProviderHealth {
  name: string;
  configured: boolean;
  model: string;
  healthy: boolean;
  cooldown_remaining_seconds: number;
  last_probe: string | null;
}

/**
 * What the service can currently do.
 *
 * "cache_only" is the state the deployment is most likely to be found in: the
 * data layer is healthy and the evaluation questions answer instantly, but no
 * model provider is reachable so new questions cannot be planned.
 */
export type ServiceMode = "full" | "cache_only" | "offline";

/** Service readiness, from GET /api/health. */
export interface HealthResponse {
  status: "ok" | "degraded";
  database_ready: boolean;
  fact_orders_rows: number | null;
  orchestrator_ready: boolean;
  providers: ProviderHealth[];
  providers_configured: string[];
  providers_healthy: string[];
  cached_answers: number;
  degraded: boolean;
  mode: ServiceMode;
  data_asof: string;
  environment: string;
  version: string;
  database_error: string | null;
}

/** One check from the data quality gate. */
export interface QualityCheck {
  name: string;
  category: string;
  severity: CheckSeverity;
  passed: boolean;
  message: string;
  details: Record<string, unknown> | null;
}

/** The data quality gate's report, from GET /api/verify. */
export interface DataQualityResponse {
  passed: boolean;
  total_checks: number;
  error_count: number;
  warning_count: number;
  info_count: number;
  categories: Record<string, QualityCheck[]>;
  checks: QualityCheck[];
}

/** The semantic layer, from GET /api/schema. */
export interface SchemaResponse {
  compact_schema: string;
  metrics: string[];
  tables: string[];
  data_asof: string;
  data_start: string;
  revenue_metric: string;
}

/** A structured failure from the API. Never a traceback. */
export interface ErrorPayload {
  error: string;
  message: string;
  request_id: string;
  detail: unknown;
}
