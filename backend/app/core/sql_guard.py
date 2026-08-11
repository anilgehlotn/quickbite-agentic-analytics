"""Validation and sandboxed execution for agent-generated SQL.

This is a security boundary, not a linter. The system is publicly deployed and
an LLM writes the queries, so any text a user types reaches a SQL generator:
prompt injection is an expected input, not a hypothetical one.

Validation is **parse-based**, never regex. Regex over SQL loses to comments,
string literals, nested quoting and whitespace — ``SELECT 1 -- ' \\n; DROP TABLE
x`` defeats a naive pattern but not a parser, which sees two statements. Every
check here inspects the ``sqlglot`` AST.

The defences are layered so that no single one is load-bearing:

1. The SQL must parse as exactly one SQLite statement.
2. That statement must be a query type (``SELECT``, or a set operation). This is
   a check on the parsed node class, so ``DROP`` cannot be disguised.
3. Every table it reads must be on the semantic layer's allowlist, with CTEs
   defined in the same query resolved first so they are not mistaken for
   unknown tables.
4. No ``sqlite_*`` internal table may be touched.
5. A ``LIMIT`` is required, and injected when missing.
6. Execution happens over a **read-only URI connection**, so even a query that
   somehow passed every check above physically cannot write.

Layer 6 is the one that matters most: layers 1-5 are software that can have
bugs, while a connection opened ``mode=ro`` is enforced by SQLite itself.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.config import settings
from app.core.logging import get_logger
from app.semantic.schema import TABLE_ALLOWLIST

logger = get_logger(__name__)

# SQLite is the target dialect; parsing as anything else would accept syntax
# the database will reject.
SQL_DIALECT: Final[str] = "sqlite"

# Prefix of SQLite's internal tables. sqlite_master exposes the schema of every
# table including ones outside the allowlist, so it is refused outright.
SQLITE_INTERNAL_PREFIX: Final[str] = "sqlite_"

# A legitimate analytical query over this schema needs at most a handful of
# joins. Far more suggests either a confused agent or a deliberate attempt to
# make the planner do expensive work.
MAX_JOINS: Final[int] = 10

# How often the progress handler runs, in SQLite virtual-machine instructions.
# Small enough that the deadline is checked promptly, large enough that the
# callback is not itself the bottleneck.
PROGRESS_HANDLER_INSTRUCTIONS: Final[int] = 1_000


class QueryExecutionError(RuntimeError):
    """A query that could not be executed.

    The message is fed back to the SQL agent so it can repair its own query,
    so it carries the failing SQL rather than only the database's complaint.

    Attributes:
        sql: The SQL that failed.
        reason: The underlying database error message.
    """

    def __init__(self, reason: str, sql: str) -> None:
        """Initialise the error.

        Args:
            reason: The underlying failure, usually the SQLite message.
            sql: The SQL that produced it.
        """
        super().__init__(f"{reason} | SQL: {sql}")
        self.reason = reason
        self.sql = sql


@dataclass
class ValidationResult:
    """The outcome of validating one statement.

    Attributes:
        valid: Whether the SQL is safe to execute.
        sql: The SQL to run, which may differ from the input when a LIMIT was
            injected. Unchanged when validation failed.
        errors: Reasons the SQL was rejected. Empty when valid.
        warnings: Non-fatal observations, such as an injected LIMIT.
        tables_referenced: Allowlisted tables the query reads, excluding CTEs.
    """

    valid: bool
    sql: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tables_referenced: list[str] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        """The errors as one string, for feeding back to an agent.

        Returns:
            Semicolon-separated errors, or an empty string when valid.
        """
        return "; ".join(self.errors)


class SQLGuard:
    """Validates agent-generated SQL against the allowlist and safety rules."""

    def __init__(
        self,
        allowed_tables: list[str] | None = None,
        max_rows: int | None = None,
        max_joins: int = MAX_JOINS,
    ) -> None:
        """Initialise the guard.

        Args:
            allowed_tables: Tables a query may read. Defaults to the semantic
                layer's allowlist, so the guard and the agents' prompts can
                never disagree about what exists.
            max_rows: Row cap injected when a query has no LIMIT. Defaults to
                ``settings.MAX_QUERY_ROWS``.
            max_joins: Maximum joins permitted, as a complexity guard.
        """
        self.allowed_tables = set(allowed_tables or TABLE_ALLOWLIST)
        self.max_rows = max_rows if max_rows is not None else settings.MAX_QUERY_ROWS
        self.max_joins = max_joins

    def validate(self, sql: str) -> ValidationResult:
        """Validate one SQL statement.

        Args:
            sql: The candidate SQL.

        Returns:
            The validation outcome. When valid, ``sql`` is the statement to
            execute and may carry an injected LIMIT.
        """
        text = (sql or "").strip()
        if not text:
            return ValidationResult(
                valid=False, sql=sql, errors=["empty SQL statement"]
            )

        # 1. Parse. Anything the SQLite parser rejects is rejected here, which
        #    also catches statements sqlglot has no node for (ATTACH, REPLACE).
        try:
            statements = sqlglot.parse(text, read=SQL_DIALECT)
        except ParseError as error:
            return ValidationResult(
                valid=False,
                sql=sql,
                errors=[f"SQL does not parse as SQLite: {error}"],
            )

        statements = [statement for statement in statements if statement is not None]

        # 2. Exactly one statement. A trailing statement after a semicolon is
        #    the classic injection payload and is counted here, not matched.
        if not statements:
            return ValidationResult(
                valid=False, sql=sql, errors=["no SQL statement found"]
            )
        if len(statements) > 1:
            kinds = ", ".join(type(s).__name__ for s in statements)
            return ValidationResult(
                valid=False,
                sql=sql,
                errors=[
                    f"expected exactly one statement, found {len(statements)} "
                    f"({kinds}); multiple statements are never permitted"
                ],
            )

        statement = statements[0]
        errors: list[str] = []
        warnings: list[str] = []

        # 3. Must be a read. Checked on the node class, so no amount of
        #    comment or whitespace trickery disguises a write.
        if not isinstance(statement, exp.Query):
            return ValidationResult(
                valid=False,
                sql=sql,
                errors=[
                    f"only SELECT statements are permitted, got "
                    f"{type(statement).__name__.upper()}"
                ],
            )

        # 4. Tables. CTE names resolve within the query and are not real
        #    tables, so they are subtracted before the allowlist check.
        cte_names = {
            cte.alias_or_name.lower()
            for cte in statement.find_all(exp.CTE)
            if cte.alias_or_name
        }
        referenced = {
            table.name.lower() for table in statement.find_all(exp.Table) if table.name
        }
        real_tables = referenced - cte_names

        internal = sorted(
            name for name in real_tables if name.startswith(SQLITE_INTERNAL_PREFIX)
        )
        if internal:
            errors.append(
                f"access to SQLite internal table(s) {internal} is not permitted"
            )

        unknown = sorted(
            name
            for name in real_tables
            if name not in self.allowed_tables
            and not name.startswith(SQLITE_INTERNAL_PREFIX)
        )
        if unknown:
            errors.append(
                f"unknown table(s) {unknown}; allowed tables are "
                f"{sorted(self.allowed_tables)}"
            )

        # 5. Complexity guard.
        join_count = len(list(statement.find_all(exp.Join)))
        if join_count > self.max_joins:
            errors.append(
                f"query has {join_count} joins, more than the {self.max_joins} "
                f"permitted"
            )

        if errors:
            return ValidationResult(
                valid=False,
                sql=sql,
                errors=errors,
                warnings=warnings,
                tables_referenced=sorted(real_tables & self.allowed_tables),
            )

        # 6. Row cap. An unbounded query can exhaust memory on a free-tier
        #    host, so a missing LIMIT is added rather than treated as an error.
        final_sql = statement.sql(dialect=SQL_DIALECT)
        if statement.args.get("limit") is None:
            final_sql = statement.limit(self.max_rows).sql(dialect=SQL_DIALECT)
            warnings.append(f"no LIMIT present; LIMIT {self.max_rows} was added")

        return ValidationResult(
            valid=True,
            sql=final_sql,
            errors=[],
            warnings=warnings,
            tables_referenced=sorted(real_tables),
        )


def to_json_safe(value: Any) -> Any:
    """Convert a SQLite value to something JSON can carry.

    Args:
        value: A value from a result row.

    Returns:
        The value unchanged when already JSON-native, otherwise a string.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


@dataclass(frozen=True)
class ExecutionResult:
    """The outcome of running one validated query.

    Attributes:
        sql: The exact SQL that was executed, including any injected LIMIT.
        columns: Result column names, in order.
        rows: Result rows as JSON-safe mappings.
        row_count: Number of rows returned after any truncation.
        execution_ms: Wall-clock execution time in milliseconds.
        truncated: Whether rows were dropped to respect the row cap.
    """

    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    execution_ms: float
    truncated: bool


class SafeQueryExecutor:
    """Runs validated SQL against a read-only connection with a deadline."""

    def __init__(
        self,
        db_path: Path | None = None,
        guard: SQLGuard | None = None,
        timeout_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        """Initialise the executor.

        Args:
            db_path: Database to query. Defaults to ``settings.DB_PATH``.
            guard: Validator to re-run before execution. Defaults to a fresh
                :class:`SQLGuard`.
            timeout_seconds: Deadline for a single query. Defaults to
                ``settings.QUERY_TIMEOUT_SECONDS``.
            max_rows: Row cap. Defaults to ``settings.MAX_QUERY_ROWS``.
        """
        self.db_path = db_path or settings.DB_PATH
        self.guard = guard or SQLGuard(max_rows=max_rows)
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.QUERY_TIMEOUT_SECONDS
        )
        self.max_rows = max_rows if max_rows is not None else settings.MAX_QUERY_ROWS

    def connect(self) -> sqlite3.Connection:
        """Open a read-only connection to the database.

        The ``mode=ro`` URI is the last line of defence: SQLite itself refuses
        writes on this handle, so a query that slipped past every validation
        check still cannot modify the data.

        Returns:
            An open read-only connection.

        Raises:
            QueryExecutionError: If the database file is missing or cannot be
                opened.
        """
        if not self.db_path.exists():
            raise QueryExecutionError(
                f"database not found at {self.db_path}", sql=""
            )
        try:
            connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        except sqlite3.Error as error:
            raise QueryExecutionError(
                f"could not open database read-only: {error}", sql=""
            ) from error
        connection.row_factory = sqlite3.Row
        return connection

    def execute(self, sql: str, validate: bool = True) -> ExecutionResult:
        """Validate and run a query, never letting a raw sqlite3 error escape.

        Args:
            sql: The SQL to run.
            validate: Whether to re-validate before executing. Left on by
                default: the caller has usually validated already, but a second
                pass costs one parse and removes any way for unvalidated SQL to
                reach the database.

        Returns:
            The result rows, columns and timing.

        Raises:
            QueryExecutionError: If validation fails, the deadline is exceeded,
                or the database rejects the query. The message carries the SQL
                because it is fed back to the agent for self-correction.
        """
        statement = sql
        if validate:
            result = self.guard.validate(sql)
            if not result.valid:
                raise QueryExecutionError(
                    f"SQL failed validation: {result.error_message}", sql=sql
                )
            statement = result.sql

        connection = self.connect()
        deadline = time.monotonic() + self.timeout_seconds

        def abort_if_overdue() -> int:
            """Abort the query once the deadline passes.

            Returns:
                1 to abort, 0 to continue.
            """
            return 1 if time.monotonic() > deadline else 0

        started = time.perf_counter()
        try:
            connection.set_progress_handler(
                abort_if_overdue, PROGRESS_HANDLER_INSTRUCTIONS
            )
            cursor = connection.execute(statement)
            fetched = cursor.fetchmany(self.max_rows + 1)
            columns = [description[0] for description in (cursor.description or [])]
        except sqlite3.OperationalError as error:
            elapsed = time.perf_counter() - started
            if time.monotonic() > deadline:
                raise QueryExecutionError(
                    f"query exceeded the {self.timeout_seconds}s timeout "
                    f"(ran for {elapsed:.1f}s) and was aborted",
                    sql=statement,
                ) from error
            raise QueryExecutionError(str(error), sql=statement) from error
        except sqlite3.Error as error:
            raise QueryExecutionError(
                f"{type(error).__name__}: {error}", sql=statement
            ) from error
        finally:
            connection.set_progress_handler(None, 0)
            connection.close()

        execution_ms = (time.perf_counter() - started) * 1000

        truncated = len(fetched) > self.max_rows
        rows = [
            {key: to_json_safe(row[key]) for key in row.keys()}
            for row in fetched[: self.max_rows]
        ]
        if truncated:
            logger.warning(
                "query_truncated",
                extra={"max_rows": self.max_rows, "sql": statement[:200]},
            )

        logger.info(
            "query_executed",
            extra={
                "row_count": len(rows),
                "execution_ms": round(execution_ms, 2),
                "truncated": truncated,
            },
        )
        return ExecutionResult(
            sql=statement,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            execution_ms=execution_ms,
            truncated=truncated,
        )
