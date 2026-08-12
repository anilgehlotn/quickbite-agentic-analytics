"use client";

/**
 * Renders a query result according to the ChartSpec the insight agent chose.
 *
 * The component is defensive by design. The spec is produced by a language
 * model, and although the backend replaces a spec whose columns do not exist,
 * this is the last place a malformed spec could take the page down. Every path
 * that cannot produce a meaningful chart returns null rather than throwing: a
 * missing chart costs a reader nothing, a blank screen costs them everything.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  formatCell,
  formatInrCompact,
  humaniseColumn,
  isMoneyColumn,
} from "@/lib/format";
import type { CellValue, ChartSpec, QueryResult, ResultRow } from "@/lib/types";

/** Chart height in pixels. Tall enough to read, short enough to scroll past. */
const CHART_HEIGHT = 280;

/** Maximum categories plotted before a chart becomes unreadable. */
const MAX_CATEGORIES = 24;

/** Maximum series in a grouped chart. Beyond this the legend dominates. */
const MAX_SERIES = 8;

/**
 * Palette for series: the accent first, muted neutrals after it.
 *
 * These are the same CSS variables the rest of the interface uses. SVG resolves
 * `var()` in `fill` and `stroke`, so the chart cannot drift out of step with
 * the page — there is one palette, declared once, in app/globals.css.
 */
const SERIES_COLOURS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
  "var(--chart-6)",
  "var(--chart-7)",
  "var(--chart-8)",
];

/** Axis, grid and hover colours, from the same palette. */
const AXIS_COLOUR = "var(--color-faint)";
const GRID_COLOUR = "var(--color-border)";
const CURSOR_COLOUR = "var(--color-accent-soft)";

/**
 * A row shaped for Recharts: one x value plus one numeric field per series.
 */
type ChartRow = Record<string, string | number>;

/**
 * Format an axis tick, abbreviating money and leaving everything else plain.
 *
 * @param value - The tick value.
 * @param money - Whether the axis carries a monetary amount.
 * @returns The tick label.
 */
function formatTick(value: number, money: boolean): string {
  if (money) {
    return formatInrCompact(value);
  }
  if (Math.abs(value) >= 1_000) {
    return `${(value / 1_000).toFixed(1)}k`;
  }
  return value.toLocaleString("en-IN", { maximumFractionDigits: 1 });
}

/**
 * Coerce a cell to a number when it is one.
 *
 * @param value - The cell value.
 * @returns The number, or null when the cell is not numeric.
 */
function toNumber(value: CellValue): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Coerce a cell to an axis label.
 *
 * @param value - The cell value.
 * @returns A string label; an em dash for null.
 */
function toLabel(value: CellValue): string {
  if (value === null) {
    return "—";
  }
  return typeof value === "string" ? value : String(value);
}

/**
 * The tooltip shown on hover, formatted with the same units as the table.
 *
 * @param props.active - Whether the tooltip is visible.
 * @param props.payload - The hovered points.
 * @param props.label - The x value under the cursor.
 * @returns The rendered tooltip, or null when nothing is hovered.
 */
function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number | string; color?: string }>;
  label?: string | number;
}): JSX.Element | null {
  if (active !== true || payload === undefined || payload.length === 0) {
    return null;
  }
  return (
    <div className="rounded border border-border bg-surface px-3 py-2 shadow-pop">
      <p className="text-label font-semibold text-ink">{String(label ?? "")}</p>
      <ul className="mt-1.5 space-y-1">
        {payload.map((entry, index) => (
          <li
            key={`${entry.name ?? index}`}
            className="flex items-center gap-2 text-label text-muted"
          >
            <span
              aria-hidden="true"
              className="h-1.5 w-1.5 shrink-0 rounded-full"
              style={{ backgroundColor: entry.color ?? SERIES_COLOURS[0] }}
            />
            <span>{humaniseColumn(entry.name ?? "")}</span>
            <span className="ml-auto pl-4 font-medium tabular-nums text-ink">
              {typeof entry.value === "number"
                ? formatCell(entry.name ?? "", entry.value)
                : String(entry.value ?? "")}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Pick the result the chart spec refers to.
 *
 * The spec names columns but not which sub-query produced them, so the result
 * is identified by having all of them.
 *
 * @param results - Every sub-query result.
 * @param spec - The chart specification.
 * @returns The matching result, or null when no result carries those columns.
 */
function findResult(
  results: QueryResult[],
  spec: ChartSpec,
): QueryResult | null {
  const needed = [spec.x_field, ...spec.y_fields].filter(
    (field) => field.length > 0,
  );
  if (needed.length === 0) {
    return null;
  }
  const matches = results.filter(
    (result) =>
      result.error === null &&
      result.rows.length > 0 &&
      needed.every((field) => result.columns.includes(field)),
  );
  if (matches.length === 0) {
    return null;
  }

  // "Top 5 and bottom 5" is two sub-queries with identical columns, and
  // charting only the first would draw five bars under a caption promising
  // ten. Results are merged only when their column sets match exactly, which
  // is what makes them halves of one ranking rather than different questions.
  const first = matches[0];
  const signature = [...first.columns].sort().join("|");
  const siblings = matches.filter(
    (result) => [...result.columns].sort().join("|") === signature,
  );
  if (siblings.length === 1) {
    return first;
  }
  return {
    ...first,
    rows: siblings.flatMap((result) => result.rows),
    row_count: siblings.reduce((total, result) => total + result.row_count, 0),
  };
}

/**
 * Reshape rows into one record per x value, with a column per series.
 *
 * @param rows - The result rows.
 * @param xField - The column plotted on the x axis.
 * @param valueField - The column carrying the measure.
 * @param seriesField - The column that splits the data into series.
 * @returns The pivoted rows and the series names, capped for legibility.
 */
function pivotBySeries(
  rows: ResultRow[],
  xField: string,
  valueField: string,
  seriesField: string,
): { data: ChartRow[]; series: string[] } {
  const byX = new Map<string, ChartRow>();
  const series: string[] = [];

  for (const row of rows) {
    const x = toLabel(row[xField]);
    const name = toLabel(row[seriesField]);
    const value = toNumber(row[valueField]);
    if (value === null) {
      continue;
    }
    if (!series.includes(name)) {
      if (series.length >= MAX_SERIES) {
        continue;
      }
      series.push(name);
    }
    const existing = byX.get(x) ?? { __x: x };
    existing[name] = value;
    byX.set(x, existing);
  }

  return {
    data: Array.from(byX.values()).slice(0, MAX_CATEGORIES),
    series,
  };
}

/**
 * Build flat chart rows for a single-series chart.
 *
 * @param rows - The result rows.
 * @param xField - The column plotted on the x axis.
 * @param yFields - The measure columns.
 * @returns One record per row, with the x value under `__x`.
 */
function flatten(
  rows: ResultRow[],
  xField: string,
  yFields: string[],
): ChartRow[] {
  const data: ChartRow[] = [];
  for (const row of rows.slice(0, MAX_CATEGORIES)) {
    const record: ChartRow = { __x: toLabel(row[xField]) };
    let hasValue = false;
    for (const field of yFields) {
      const value = toNumber(row[field]);
      if (value !== null) {
        record[field] = value;
        hasValue = true;
      }
    }
    if (hasValue) {
      data.push(record);
    }
  }
  return data;
}

/**
 * Render the chart described by a ChartSpec, or nothing when it cannot be.
 *
 * @param props.spec - The chart specification, or null.
 * @param props.results - Every sub-query result.
 * @returns The chart, or null when there is nothing sensible to draw.
 */
export default function ResultChart({
  spec,
  results,
}: {
  spec: ChartSpec | null;
  results: QueryResult[];
}): JSX.Element | null {
  if (spec === null || spec.chart_type === "none") {
    return null;
  }

  const result = findResult(results, spec);
  if (result === null) {
    return null;
  }

  const yFields = spec.y_fields.filter((field) =>
    result.columns.includes(field),
  );
  if (yFields.length === 0) {
    return null;
  }

  const grouped =
    spec.series_field !== null &&
    spec.series_field.length > 0 &&
    result.columns.includes(spec.series_field);

  const { data, series } = grouped
    ? pivotBySeries(
        result.rows,
        spec.x_field,
        yFields[0],
        spec.series_field as string,
      )
    : { data: flatten(result.rows, spec.x_field, yFields), series: yFields };

  if (data.length === 0 || series.length === 0) {
    return null;
  }

  const money = yFields.some((field) => isMoneyColumn(field));
  const axisProps = {
    stroke: GRID_COLOUR,
    tick: { fill: AXIS_COLOUR, fontSize: 11 },
    tickLine: false,
  } as const;

  const isLine = spec.chart_type === "line";

  return (
    <figure className="mt-8">
      <figcaption className="label-caps mb-4">{spec.title}</figcaption>
      <div style={{ width: "100%", height: CHART_HEIGHT }}>
        <ResponsiveContainer width="100%" height="100%">
          {isLine ? (
            <LineChart
              data={data}
              margin={{ top: 8, right: 12, bottom: 4, left: 4 }}
            >
              <CartesianGrid stroke={GRID_COLOUR} vertical={false} />
              <XAxis dataKey="__x" {...axisProps} />
              <YAxis
                {...axisProps}
                width={64}
                tickFormatter={(value: number) => formatTick(value, money)}
              />
              <Tooltip
                content={<ChartTooltip />}
                cursor={{ stroke: GRID_COLOUR }}
              />
              {series.length > 1 && (
                <Legend
                  formatter={(value: string) => (
                    <span className="text-label text-muted">
                      {grouped ? value : humaniseColumn(value)}
                    </span>
                  )}
                />
              )}
              {series.map((name, index) => (
                <Line
                  key={name}
                  type="monotone"
                  dataKey={name}
                  stroke={SERIES_COLOURS[index % SERIES_COLOURS.length]}
                  strokeWidth={1.75}
                  dot={{ r: 2 }}
                  activeDot={{ r: 4 }}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          ) : (
            <BarChart
              data={data}
              margin={{ top: 8, right: 12, bottom: 4, left: 4 }}
            >
              <CartesianGrid stroke={GRID_COLOUR} vertical={false} />
              <XAxis
                dataKey="__x"
                {...axisProps}
                interval={0}
                angle={data.length > 6 ? -35 : 0}
                textAnchor={data.length > 6 ? "end" : "middle"}
                height={data.length > 6 ? 72 : 32}
              />
              <YAxis
                {...axisProps}
                width={64}
                tickFormatter={(value: number) => formatTick(value, money)}
              />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: CURSOR_COLOUR }} />
              {series.length > 1 && (
                <Legend
                  formatter={(value: string) => (
                    <span className="text-label text-muted">
                      {grouped ? value : humaniseColumn(value)}
                    </span>
                  )}
                />
              )}
              {series.map((name, index) => (
                <Bar
                  key={name}
                  dataKey={name}
                  fill={SERIES_COLOURS[index % SERIES_COLOURS.length]}
                  radius={[2, 2, 0, 0]}
                  isAnimationActive={false}
                />
              ))}
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
