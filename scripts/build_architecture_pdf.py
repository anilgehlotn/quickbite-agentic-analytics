"""Render docs/ARCHITECTURE.md into docs/Agent_Architecture.pdf.

Uses ReportLab, which is pure Python and needs no network access, no headless
browser and no system libraries. That constraint is the reason for two
decisions:

* **Markdown is parsed here rather than by a library.** The document uses a
  small, known subset - headings, paragraphs, bullet and numbered lists, fenced
  code, tables, block quotes, and inline bold/italic/code/links. Handling that
  subset directly is about two hundred lines and removes a dependency whose
  HTML output would then need a second converter anyway.
* **The Mermaid diagram is drawn as vectors, not rendered.** Rendering Mermaid
  offline needs Node and a headless browser. Instead the ``mermaid`` fence is
  replaced by a ReportLab drawing of the same pipeline, which is sharper than a
  rasterised image, scales with the page, and adds no tooling. The document's
  own text fallback is kept as well, so the PDF carries both.

Usage::

    python scripts/build_architecture_pdf.py
    python scripts/build_architecture_pdf.py --source docs/OTHER.md --out x.pdf
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE: Final[Path] = _ROOT / "docs" / "ARCHITECTURE.md"
DEFAULT_OUTPUT: Final[Path] = _ROOT / "docs" / "Agent_Architecture.pdf"

PROJECT_NAME: Final[str] = "QuickBite Agentic Analytics"
REPOSITORY_URL: Final[str] = (
    "https://github.com/anilgehlotn/quickbite-agentic-analytics"
)
DOCUMENT_TITLE: Final[str] = "Agent Architecture"

# Palette, matching the application interface so the document and the product
# look like one thing.
INK: Final[colors.Color] = colors.HexColor("#1A1917")
MUTED: Final[colors.Color] = colors.HexColor("#57534E")
FAINT: Final[colors.Color] = colors.HexColor("#6F6963")
ACCENT: Final[colors.Color] = colors.HexColor("#4C2A4D")
ACCENT_SOFT: Final[colors.Color] = colors.HexColor("#F5EFF4")
RULE: Final[colors.Color] = colors.HexColor("#D5D2CB")
HAIRLINE: Final[colors.Color] = colors.HexColor("#E7E5E1")
RAISED: Final[colors.Color] = colors.HexColor("#F4F3EF")

PAGE_MARGIN: Final[float] = 20 * mm
BODY_WIDTH: Final[float] = A4[0] - 2 * PAGE_MARGIN


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def build_styles() -> dict[str, ParagraphStyle]:
    """Create the paragraph styles used throughout the document.

    Returns:
        Style name to style.
    """
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14.5,
        textColor=INK,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=7,
    )
    return {
        "body": body,
        "h1": ParagraphStyle(
            "H1", parent=body, fontName="Helvetica-Bold", fontSize=19,
            leading=24, textColor=INK, spaceBefore=0, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2", parent=body, fontName="Helvetica-Bold", fontSize=14.5,
            leading=19, textColor=INK, spaceBefore=16, spaceAfter=8,
            keepWithNext=1,
        ),
        "h3": ParagraphStyle(
            "H3", parent=body, fontName="Helvetica-Bold", fontSize=11,
            leading=15, textColor=ACCENT, spaceBefore=12, spaceAfter=5,
            keepWithNext=1,
        ),
        "h4": ParagraphStyle(
            "H4", parent=body, fontName="Helvetica-BoldOblique", fontSize=10,
            leading=14, textColor=MUTED, spaceBefore=9, spaceAfter=4,
            keepWithNext=1,
        ),
        "code": ParagraphStyle(
            "Code", parent=body, fontName="Courier", fontSize=7.8, leading=10.5,
            textColor=INK, spaceBefore=0, spaceAfter=0,
        ),
        "quote": ParagraphStyle(
            "Quote", parent=body, fontName="Helvetica-Oblique", fontSize=10,
            leading=15, textColor=ACCENT, leftIndent=8, spaceBefore=4,
            spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            "Cell", parent=body, fontSize=8, leading=11, spaceAfter=0,
        ),
        "cellhead": ParagraphStyle(
            "CellHead", parent=body, fontName="Helvetica-Bold", fontSize=8,
            leading=11, spaceAfter=0, textColor=INK,
        ),
        "title": ParagraphStyle(
            "Title", parent=body, fontName="Helvetica-Bold", fontSize=30,
            leading=35, textColor=INK, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=body, fontName="Helvetica", fontSize=13,
            leading=18, textColor=MUTED, spaceAfter=3,
        ),
        "meta": ParagraphStyle(
            "Meta", parent=body, fontName="Helvetica", fontSize=9,
            leading=13, textColor=FAINT, spaceAfter=2,
        ),
    }


# ---------------------------------------------------------------------------
# Inline markdown
# ---------------------------------------------------------------------------

_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*]+)\*(?!\*)")


def inline(text: str) -> str:
    """Convert inline markdown to ReportLab's mini-HTML.

    Code spans are escaped and substituted before anything else, so markdown
    characters *inside* a code span cannot be interpreted as formatting.

    Args:
        text: One paragraph of markdown source.

    Returns:
        Markup suitable for a ReportLab Paragraph.
    """
    spans: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        spans.append(match.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = _CODE.sub(stash_code, text)
    text = html.escape(text, quote=False)
    text = _LINK.sub(
        lambda m: f'<link href="{html.escape(m.group(2), quote=True)}" '
        f'color="#4C2A4D">{m.group(1)}</link>',
        text,
    )
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    # Restore code spans last, escaped and monospaced.
    for index, span in enumerate(spans):
        text = text.replace(
            f"\x00{index}\x00",
            f'<font face="Courier" size="8.5" color="#4C2A4D">'
            f"{html.escape(span, quote=False)}</font>",
        )
    return text


# ---------------------------------------------------------------------------
# The pipeline diagram, drawn rather than rendered
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """One node in the pipeline diagram.

    Attributes:
        label: Text inside the box.
        sub: Smaller second line, or empty.
        kind: ``"agent"``, ``"infra"`` or ``"terminal"``, deciding the fill.
    """

    label: str
    sub: str
    kind: str


class PipelineDiagram(Flowable):
    """A vector drawing of the agent pipeline.

    Drawn with ReportLab primitives so it needs no Mermaid toolchain, stays
    sharp at any zoom, and cannot fail at build time for want of a browser.
    """

    #: The vertical chain, top to bottom.
    CHAIN: Final[tuple[Box, ...]] = (
        Box("User question", "", "terminal"),
        Box("Entity resolver", "deterministic, no model", "infra"),
        Box("Planner agent", "emits AnalysisPlan", "agent"),
        Box("SQL analyst agent", "one query per sub-query, concurrent", "agent"),
        Box("SQL guard", "parse, allowlist, LIMIT, read-only", "infra"),
        Box("SQLite", "star schema, mode=ro", "infra"),
        Box("Verifier agent", "13 deterministic checks", "agent"),
        Box("Insight agent", "explanation and chart spec", "agent"),
        Box("Response", "answer, verification, full trace", "terminal"),
    )

    BOX_WIDTH: Final[float] = 78 * mm
    BOX_HEIGHT: Final[float] = 12.5 * mm
    GAP: Final[float] = 7 * mm
    SIDE_WIDTH: Final[float] = 42 * mm

    def __init__(self) -> None:
        """Size the flowable to fit its content."""
        super().__init__()
        self.width = BODY_WIDTH
        self.height = len(self.CHAIN) * self.BOX_HEIGHT + (
            len(self.CHAIN) - 1
        ) * self.GAP

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:
        """Report the space required.

        Args:
            availWidth: Width offered by the frame.
            availHeight: Height offered by the frame.

        Returns:
            The width and height this flowable will occupy.
        """
        return self.width, self.height

    def _box(
        self, canvas: Canvas, x: float, y: float, box: Box, width: float
    ) -> None:
        """Draw one labelled box.

        Args:
            canvas: The target canvas.
            x: Left edge.
            y: Bottom edge.
            box: The node to draw.
            width: Box width.
        """
        fills = {
            "agent": ACCENT_SOFT,
            "infra": RAISED,
            "terminal": colors.white,
        }
        strokes = {"agent": ACCENT, "infra": RULE, "terminal": INK}
        canvas.setFillColor(fills[box.kind])
        canvas.setStrokeColor(strokes[box.kind])
        canvas.setLineWidth(0.9 if box.kind == "agent" else 0.6)
        canvas.roundRect(x, y, width, self.BOX_HEIGHT, 2.2, stroke=1, fill=1)

        canvas.setFillColor(INK)
        if box.sub:
            canvas.setFont("Helvetica-Bold", 8.6)
            canvas.drawCentredString(
                x + width / 2, y + self.BOX_HEIGHT - 5.4 * mm, box.label
            )
            canvas.setFont("Helvetica", 6.9)
            canvas.setFillColor(FAINT)
            canvas.drawCentredString(
                x + width / 2, y + self.BOX_HEIGHT - 9.2 * mm, box.sub
            )
        else:
            canvas.setFont("Helvetica-Bold", 9.2)
            canvas.drawCentredString(
                x + width / 2, y + self.BOX_HEIGHT / 2 - 1.1 * mm, box.label
            )

    def _arrow(self, canvas: Canvas, x: float, top: float, bottom: float) -> None:
        """Draw a downward arrow between two boxes.

        Args:
            canvas: The target canvas.
            x: Horizontal centre.
            top: Y of the arrow's tail.
            bottom: Y of the arrow's head.
        """
        canvas.setStrokeColor(MUTED)
        canvas.setFillColor(MUTED)
        canvas.setLineWidth(0.8)
        canvas.line(x, top, x, bottom + 1.6 * mm)
        path = canvas.beginPath()
        path.moveTo(x, bottom)
        path.lineTo(x - 1.3 * mm, bottom + 2.1 * mm)
        path.lineTo(x + 1.3 * mm, bottom + 2.1 * mm)
        path.close()
        canvas.drawPath(path, stroke=0, fill=1)

    def draw(self) -> None:
        """Render the diagram onto the canvas."""
        canvas = self.canv
        centre = self.width / 2
        left = centre - self.BOX_WIDTH / 2
        step = self.BOX_HEIGHT + self.GAP

        for index, box in enumerate(self.CHAIN):
            y = self.height - (index + 1) * self.BOX_HEIGHT - index * self.GAP
            self._box(canvas, left, y, box, self.BOX_WIDTH)
            if index < len(self.CHAIN) - 1:
                self._arrow(canvas, centre, y, y - self.GAP)

        # Semantic layer feeds the planner and the analyst (dotted, from the
        # left). Indices 2 and 3 in the chain.
        canvas.setDash(1.6, 1.6)
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.7)
        side_x = left - self.SIDE_WIDTH - 6 * mm
        planner_y = self.height - 3 * self.BOX_HEIGHT - 2 * self.GAP
        analyst_y = self.height - 4 * self.BOX_HEIGHT - 3 * self.GAP
        side_y = (planner_y + analyst_y) / 2
        canvas.setFillColor(colors.white)
        canvas.setStrokeColor(RULE)
        canvas.setDash()
        canvas.roundRect(
            side_x, side_y, self.SIDE_WIDTH, self.BOX_HEIGHT, 2.2, stroke=1, fill=1
        )
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 7.8)
        canvas.drawCentredString(
            side_x + self.SIDE_WIDTH / 2, side_y + self.BOX_HEIGHT - 5.0 * mm,
            "Semantic layer",
        )
        canvas.setFont("Helvetica", 6.6)
        canvas.setFillColor(FAINT)
        canvas.drawCentredString(
            side_x + self.SIDE_WIDTH / 2, side_y + self.BOX_HEIGHT - 8.6 * mm,
            "schema, metrics, rules",
        )
        canvas.setDash(1.6, 1.6)
        canvas.setStrokeColor(RULE)
        for target_y in (planner_y, analyst_y):
            canvas.line(
                side_x + self.SIDE_WIDTH,
                side_y + self.BOX_HEIGHT / 2,
                left,
                target_y + self.BOX_HEIGHT / 2,
            )
        canvas.setDash()

        # Self-healing retry: verifier back up to the analyst, on the right.
        verifier_y = self.height - 7 * self.BOX_HEIGHT - 6 * self.GAP
        right_x = left + self.BOX_WIDTH
        loop_x = right_x + 12 * mm
        canvas.setStrokeColor(ACCENT)
        canvas.setFillColor(ACCENT)
        canvas.setLineWidth(0.8)
        path = canvas.beginPath()
        path.moveTo(right_x, verifier_y + self.BOX_HEIGHT / 2)
        path.lineTo(loop_x, verifier_y + self.BOX_HEIGHT / 2)
        path.lineTo(loop_x, analyst_y + self.BOX_HEIGHT / 2)
        path.lineTo(right_x + 1.8 * mm, analyst_y + self.BOX_HEIGHT / 2)
        canvas.drawPath(path, stroke=1, fill=0)
        head = canvas.beginPath()
        head.moveTo(right_x, analyst_y + self.BOX_HEIGHT / 2)
        head.lineTo(right_x + 2.4 * mm, analyst_y + self.BOX_HEIGHT / 2 + 1.3 * mm)
        head.lineTo(right_x + 2.4 * mm, analyst_y + self.BOX_HEIGHT / 2 - 1.3 * mm)
        head.close()
        canvas.drawPath(head, stroke=0, fill=1)
        canvas.setFont("Helvetica-Oblique", 6.6)
        canvas.saveState()
        canvas.translate(loop_x + 2.4 * mm, (verifier_y + analyst_y) / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, "self-healing retry, once")
        canvas.restoreState()


# ---------------------------------------------------------------------------
# Markdown to flowables
# ---------------------------------------------------------------------------


class MarkdownRenderer:
    """Convert the document's markdown subset into ReportLab flowables."""

    def __init__(self, styles: dict[str, ParagraphStyle]) -> None:
        """Initialise the renderer.

        Args:
            styles: The paragraph styles to use.
        """
        self.styles = styles

    def render(self, source: str) -> list[Any]:
        """Convert a whole document.

        Args:
            source: The markdown text.

        Returns:
            The flowables, in order.
        """
        flowables: list[Any] = []
        lines = source.splitlines()
        index = 0
        # The title block is generated separately, so skip the leading H1 and
        # the repository line that duplicate it.
        while index < len(lines) and not lines[index].startswith("## "):
            index += 1

        while index < len(lines):
            line = lines[index]

            if not line.strip():
                index += 1
                continue

            if line.startswith("```"):
                index = self._code_block(lines, index, flowables)
                continue

            if line.startswith("|"):
                index = self._table(lines, index, flowables)
                continue

            if line.startswith("#"):
                self._heading(line, flowables)
                index += 1
                continue

            if line.startswith("> "):
                index = self._quote(lines, index, flowables)
                continue

            if re.match(r"^[-*] ", line) or re.match(r"^\d+\. ", line):
                index = self._list(lines, index, flowables)
                continue

            if set(line.strip()) == {"-"} and len(line.strip()) >= 3:
                flowables.append(Spacer(1, 3))
                index += 1
                continue

            index = self._paragraph(lines, index, flowables)

        return flowables

    def _heading(self, line: str, out: list[Any]) -> None:
        """Append a heading.

        Args:
            line: The heading line.
            out: Flowable list appended to in place.
        """
        level = len(line) - len(line.lstrip("#"))
        text = line.lstrip("#").strip()
        if level == 2:
            out.append(Spacer(1, 4))
            out.append(Paragraph(inline(text), self.styles["h2"]))
            out.append(HorizontalRule())
        else:
            style = self.styles.get(f"h{min(level, 4)}", self.styles["h4"])
            out.append(Paragraph(inline(text), style))

    def _paragraph(self, lines: list[str], index: int, out: list[Any]) -> int:
        """Append one paragraph, joining its wrapped source lines.

        Args:
            lines: All document lines.
            index: Index of the first line.
            out: Flowable list appended to in place.

        Returns:
            Index of the first unconsumed line.
        """
        buffer: list[str] = []
        while index < len(lines) and lines[index].strip():
            line = lines[index]
            if line.startswith(("#", "|", "```", "> ")) or re.match(
                r"^([-*] |\d+\. )", line
            ):
                break
            buffer.append(line.strip())
            index += 1
        if buffer:
            out.append(Paragraph(inline(" ".join(buffer)), self.styles["body"]))
        return index

    def _quote(self, lines: list[str], index: int, out: list[Any]) -> int:
        """Append a block quote, rendered as an emphasised callout.

        Args:
            lines: All document lines.
            index: Index of the first line.
            out: Flowable list appended to in place.

        Returns:
            Index of the first unconsumed line.
        """
        buffer: list[str] = []
        while index < len(lines) and lines[index].startswith(">"):
            buffer.append(lines[index].lstrip(">").strip())
            index += 1
        text = " ".join(part for part in buffer if part)
        out.append(
            Table(
                [[Paragraph(inline(text), self.styles["quote"])]],
                colWidths=[BODY_WIDTH],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_SOFT),
                        ("LINEBEFORE", (0, 0), (0, -1), 2, ACCENT),
                        ("LEFTPADDING", (0, 0), (-1, -1), 9),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            )
        )
        out.append(Spacer(1, 7))
        return index

    def _code_block(self, lines: list[str], index: int, out: list[Any]) -> int:
        """Append a fenced code block, or the diagram for a mermaid fence.

        Args:
            lines: All document lines.
            index: Index of the opening fence.
            out: Flowable list appended to in place.

        Returns:
            Index of the first line after the closing fence.
        """
        language = lines[index][3:].strip().lower()
        index += 1
        body: list[str] = []
        while index < len(lines) and not lines[index].startswith("```"):
            body.append(lines[index])
            index += 1
        index += 1  # closing fence

        if language == "mermaid":
            # Drawn natively; see PipelineDiagram for why.
            out.append(Spacer(1, 6))
            out.append(PipelineDiagram())
            out.append(Spacer(1, 10))
            return index

        rows = [
            [Paragraph(html.escape(line, quote=False) or "&nbsp;", self.styles["code"])]
            for line in body
        ]
        if not rows:
            return index
        out.append(
            Table(
                rows,
                colWidths=[BODY_WIDTH],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), RAISED),
                        ("BOX", (0, 0), (-1, -1), 0.5, HAIRLINE),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 1),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ]
                ),
            )
        )
        out.append(Spacer(1, 9))
        return index

    def _list(self, lines: list[str], index: int, out: list[Any]) -> int:
        """Append a bullet or numbered list, including wrapped continuations.

        Args:
            lines: All document lines.
            index: Index of the first item.
            out: Flowable list appended to in place.

        Returns:
            Index of the first unconsumed line.
        """
        ordered = bool(re.match(r"^\d+\. ", lines[index]))
        items: list[str] = []
        while index < len(lines):
            line = lines[index]
            if re.match(r"^([-*] |\d+\. )", line):
                items.append(re.sub(r"^([-*] |\d+\. )", "", line).strip())
                index += 1
            elif line.startswith("  ") and line.strip() and items:
                items[-1] += " " + line.strip()
                index += 1
            elif not line.strip():
                # A blank line ends the list unless another item follows.
                lookahead = index + 1
                if lookahead < len(lines) and re.match(
                    r"^([-*] |\d+\. )", lines[lookahead]
                ):
                    index += 1
                    continue
                break
            else:
                break

        out.append(
            ListFlowable(
                [
                    ListItem(
                        Paragraph(inline(item), self.styles["body"]),
                        leftIndent=13,
                        value=number + 1 if ordered else None,
                    )
                    for number, item in enumerate(items)
                ],
                bulletType="1" if ordered else "bullet",
                bulletFontSize=8,
                bulletColor=ACCENT,
                start="1" if ordered else None,
                leftIndent=13,
            )
        )
        out.append(Spacer(1, 5))
        return index

    def _table(self, lines: list[str], index: int, out: list[Any]) -> int:
        """Append a pipe table.

        Args:
            lines: All document lines.
            index: Index of the header row.
            out: Flowable list appended to in place.

        Returns:
            Index of the first unconsumed line.
        """
        rows: list[list[str]] = []
        while index < len(lines) and lines[index].startswith("|"):
            cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
            if not all(set(cell) <= set("-: ") and cell for cell in cells):
                rows.append(cells)
            index += 1
        if not rows:
            return index

        columns = max(len(row) for row in rows)
        data = [
            [
                Paragraph(
                    inline(row[column] if column < len(row) else ""),
                    self.styles["cellhead" if number == 0 else "cell"],
                )
                for column in range(columns)
            ]
            for number, row in enumerate(rows)
        ]

        # Column widths are weighted rather than equal. In these tables the
        # first column is a short label and the last is usually the longest
        # prose, so equal widths make a five-column table a stack of narrow
        # ribbons.
        if columns == 1:
            widths = [BODY_WIDTH]
        elif columns == 2:
            widths = [BODY_WIDTH * 0.34, BODY_WIDTH * 0.66]
        elif columns == 3:
            widths = [BODY_WIDTH * 0.30, BODY_WIDTH * 0.16, BODY_WIDTH * 0.54]
        else:
            weights = [1.0] + [1.3] * (columns - 2) + [2.0]
            total = sum(weights)
            widths = [BODY_WIDTH * weight / total for weight in weights]

        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), RAISED),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
                    ("LINEBELOW", (0, 1), (-1, -2), 0.4, HAIRLINE),
                    ("BOX", (0, 0), (-1, -1), 0.5, HAIRLINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        out.append(table)
        out.append(Spacer(1, 10))
        return index


class HorizontalRule(Flowable):
    """A hairline rule under a section heading."""

    def __init__(self, width: float = BODY_WIDTH) -> None:
        """Initialise the rule.

        Args:
            width: Rule width in points.
        """
        super().__init__()
        self.width = width
        self.height = 5

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:
        """Report the space required.

        Args:
            availWidth: Width offered.
            availHeight: Height offered.

        Returns:
            Width and height.
        """
        return self.width, self.height

    def draw(self) -> None:
        """Render the rule."""
        self.canv.setStrokeColor(RULE)
        self.canv.setLineWidth(0.8)
        self.canv.line(0, 3, self.width, 3)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def title_block(styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Build the cover block.

    Args:
        styles: The paragraph styles.

    Returns:
        Flowables for the title page.
    """
    return [
        Spacer(1, 58 * mm),
        Paragraph(DOCUMENT_TITLE, styles["title"]),
        Paragraph(PROJECT_NAME, styles["subtitle"]),
        Spacer(1, 8),
        HorizontalRule(),
        Spacer(1, 10),
        Paragraph(
            "Natural-language business analytics answered by four cooperating "
            "agents that plan the analysis, write and execute SQL, verify the "
            "numbers arithmetically, and explain the result.",
            styles["body"],
        ),
        Spacer(1, 12),
        Paragraph(
            f'Repository: <link href="{REPOSITORY_URL}" color="#4C2A4D">'
            f"{REPOSITORY_URL}</link>",
            styles["meta"],
        ),
        Paragraph(
            "Source: docs/ARCHITECTURE.md &mdash; regenerate with "
            "scripts/build_architecture_pdf.py",
            styles["meta"],
        ),
        PageBreak(),
    ]


def page_furniture(canvas: Canvas, document: BaseDocTemplate) -> None:
    """Draw the running header and the page number.

    The title page is left clean; numbering starts on the first content page
    and counts from one there.

    Args:
        canvas: The page canvas.
        document: The document being built.
    """
    if canvas.getPageNumber() == 1:
        return

    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(FAINT)
    canvas.drawString(
        PAGE_MARGIN, A4[1] - PAGE_MARGIN + 5 * mm,
        f"{PROJECT_NAME} — {DOCUMENT_TITLE}",
    )
    canvas.setStrokeColor(HAIRLINE)
    canvas.setLineWidth(0.5)
    canvas.line(
        PAGE_MARGIN, A4[1] - PAGE_MARGIN + 3.4 * mm,
        A4[0] - PAGE_MARGIN, A4[1] - PAGE_MARGIN + 3.4 * mm,
    )
    canvas.drawRightString(
        A4[0] - PAGE_MARGIN, PAGE_MARGIN - 7 * mm,
        str(canvas.getPageNumber() - 1),
    )
    canvas.restoreState()


def build(source_path: Path, output_path: Path) -> tuple[int, int]:
    """Render the markdown source to a PDF.

    Args:
        source_path: The markdown document.
        output_path: Where to write the PDF.

    Returns:
        The file size in bytes and the page count.

    Raises:
        FileNotFoundError: If the source does not exist.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"source not found: {source_path}")

    styles = build_styles()
    document = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
        title=f"{DOCUMENT_TITLE} — {PROJECT_NAME}",
        author=PROJECT_NAME,
        subject="Agent architecture and engineering decisions",
    )
    frame = Frame(
        PAGE_MARGIN, PAGE_MARGIN, BODY_WIDTH,
        A4[1] - 2 * PAGE_MARGIN, id="body",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    document.addPageTemplates(
        [PageTemplate(id="main", frames=[frame], onPage=page_furniture)]
    )

    story = title_block(styles)
    story.extend(MarkdownRenderer(styles).render(source_path.read_text("utf-8")))
    document.build(story)

    return output_path.stat().st_size, document.page


def main() -> int:
    """Build the PDF and report on it.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    size, pages = build(arguments.source, arguments.out)
    megabytes = size / 1_048_576
    print(f"wrote  {arguments.out}")
    print(f"pages  {pages}")
    print(f"size   {size:,} bytes ({megabytes:.2f} MB)")
    if megabytes >= 10:
        print("FAILED: over the 10 MB limit")
        return 1
    print("diagram: drawn as vectors (no Mermaid toolchain required)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
