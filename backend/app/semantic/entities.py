"""Resolve the things a question names to the values actually in the database.

The single most likely cause of a confidently wrong answer is a reviewer naming
something in a way the data does not use. The database holds ``ST007``, but a
person types "ST7"; it holds ``Bengaluru``, but a person types "Bangalore"; it
holds ``Veg Burger 5``, but a person types "veg burgers". A model asked to
invent a filter value from any of those will produce SQL that runs perfectly
and returns nothing, or worse, returns the wrong store.

So the resolution happens here, in code, against the real dimension values, and
the canonical answers are handed to the planner as facts rather than left to be
guessed. This follows the pattern established throughout the system: compute
what the model would otherwise have to infer.

Three properties matter more than recall:

* **Never invent.** A term matching nothing resolves to nothing. Answering a
  question about a store that does not exist by silently substituting the
  nearest one is the worst failure this module could have, so the fuzzy
  threshold is set high and short tokens are excluded from fuzzy matching
  entirely.
* **Ambiguity survives.** "Gurugram" names a city *and* eight stores. Rather
  than picking, every candidate is returned so the caller can cover them all or
  ask which was meant.
* **Startup cost is paid once.** The catalogue is loaded from the database on
  first use and cached, because resolving is on the request path.

Usage::

    from app.semantic.entities import resolve

    matches = resolve("how is ST7 doing versus the Bangalore stores?")
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Iterable

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# --- Tuning ---------------------------------------------------------------

#: Minimum difflib ratio for a typo to count as the same entity. Deliberately
#: high: 0.84 accepts "Bengaluru"/"Bengalru" and rejects "Delhi"/"Deli", which
#: are different enough that guessing would be inventing.
FUZZY_THRESHOLD: Final[float] = 0.84

#: Shortest token length eligible for fuzzy matching. Below this, edit distance
#: stops being evidence: "Pune" and "Pure" differ by one character but are not
#: the same word, and three-letter tokens match almost anything.
MIN_FUZZY_LENGTH: Final[int] = 5

#: Shortest name eligible for substring matching, for the same reason.
MIN_SUBSTRING_LENGTH: Final[int] = 4

#: Confidence assigned to each match kind. Exact identity is certain; a fuzzy
#: match is a suggestion. The scores are ordered rather than calibrated - their
#: only job is to rank candidates and to mark ambiguity.
CONFIDENCE_EXACT: Final[float] = 1.0
CONFIDENCE_CASE_INSENSITIVE: Final[float] = 0.98
CONFIDENCE_NORMALISED_ID: Final[float] = 0.95
CONFIDENCE_ALIAS: Final[float] = 0.95
CONFIDENCE_SUBSTRING: Final[float] = 0.8
CONFIDENCE_FUZZY_BASE: Final[float] = 0.7

#: Two candidates within this much confidence of each other are ambiguous.
AMBIGUITY_BAND: Final[float] = 0.1

#: Maximum matches returned, so a question naming a whole dimension cannot
#: flood the planner prompt.
MAX_MATCHES: Final[int] = 24

# Words that look like entity names but are analytical vocabulary. Without this
# "Which stores had the best margin?" resolves "best" against store names by
# substring and "margin" against nothing useful.
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "all", "and", "any", "are", "average", "best", "between", "both", "but",
        "compare", "comparison", "count", "customer", "customers", "data",
        "day", "days", "did", "does", "doing", "down", "drop", "each", "every",
        "for", "from", "growth", "had", "has", "have", "high", "highest", "how",
        "into", "last", "least", "low", "lowest", "many", "margin",
        "month", "months", "most", "much", "new", "not", "off", "old", "only",
        "order", "orders", "our", "over", "per", "performance", "performing",
        "product", "products", "quarter", "rank", "ranking", "rate", "revenue",
        "sales", "segment", "share", "show", "since", "sold", "store", "stores",
        "than", "that", "the", "their", "these", "they", "this", "top", "total",
        "trend", "units", "value", "versus", "was", "week", "weeks", "were",
        "what", "when", "where", "which", "who", "why", "with", "worst", "year",
        "years",
    }
)

# City spellings people still use, mapped to the value in the data. Only cities
# actually present are resolvable; the mapping is filtered against the
# catalogue at load time so a rename in the data cannot leave a dead alias.
CITY_ALIASES: Final[dict[str, str]] = {
    "bangalore": "Bengaluru",
    "banglore": "Bengaluru",
    "blr": "Bengaluru",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "gurgaon": "Gurugram",
    "ggn": "Gurugram",
    "trivandrum": "Thiruvananthapuram",
    "poona": "Pune",
    "mysore": "Mysuru",
    "baroda": "Vadodara",
    "cochin": "Kochi",
    "pondicherry": "Puducherry",
    "new delhi": "Delhi",
    "ncr": "Delhi",
    "hyd": "Hyderabad",
}

# Channel wording that does not appear in the data but clearly means one.
CHANNEL_ALIASES: Final[dict[str, str]] = {
    "dine in": "Dine-in",
    "dinein": "Dine-in",
    "eat in": "Dine-in",
    "take away": "Takeaway",
    "takeout": "Takeaway",
    "take out": "Takeaway",
    "pickup": "Takeaway",
    "pick up": "Takeaway",
}

#: Channels that are third-party delivery aggregators. "Delivery" is a common
#: way to refer to both at once, and treating it as a single channel would be
#: wrong, so it resolves to both.
DELIVERY_CHANNELS: Final[tuple[str, ...]] = ("Swiggy", "Zomato")

# Tokens that split a question into candidate phrases.
_TOKEN = re.compile(r"[A-Za-z0-9&'\-]+")

# An identifier like ST7, st007, SKU5, sku0012.
_IDENTIFIER = re.compile(r"^(?P<prefix>[A-Za-z]{2,4})(?P<digits>\d{1,6})$")


@dataclass(frozen=True)
class Dimension:
    """One resolvable attribute of the data.

    Attributes:
        name: Human-readable dimension name, used in the planner prompt.
        table: Table the column lives in.
        column: Fully-qualified column name to filter on.
        is_identifier: Whether values are codes like ``ST007``, which get
            zero-padding normalisation rather than substring matching.
    """

    name: str
    table: str
    column: str
    is_identifier: bool = False


#: Every dimension a question can name, with the column to filter on. The
#: column names are what the planner is told to use, so they must match the
#: schema exactly.
DIMENSIONS: Final[tuple[Dimension, ...]] = (
    Dimension("store_id", "dim_store", "dim_store.store_id", is_identifier=True),
    Dimension("store_name", "dim_store", "dim_store.store_name"),
    Dimension("city", "dim_store", "dim_store.city"),
    Dimension("region", "dim_store", "dim_store.region"),
    Dimension("store_format", "dim_store", "dim_store.store_format"),
    Dimension("sku_id", "dim_product", "dim_product.sku_id", is_identifier=True),
    Dimension("sku_name", "dim_product", "dim_product.sku_name"),
    Dimension("category", "dim_product", "dim_product.category"),
    Dimension("veg_nonveg", "dim_product", "dim_product.veg_nonveg"),
    Dimension("channel", "fact_orders", "fact_orders.channel"),
    Dimension(
        "customer_segment", "dim_customer", "dim_customer.customer_segment"
    ),
    Dimension("promo_name", "dim_promotion", "dim_promotion.promo_name"),
    Dimension("promo_type", "dim_promotion", "dim_promotion.promo_type"),
    Dimension(
        "festive_period", "dim_calendar", "dim_calendar.festive_period"
    ),
    Dimension("day_name", "dim_calendar", "dim_calendar.day_name"),
    Dimension("day_type", "dim_calendar", "dim_calendar.day_type"),
)


@dataclass(frozen=True)
class EntityMatch:
    """One reference in a question resolved to a value in the data.

    Attributes:
        text: The words from the question that produced this match.
        value: The canonical value, exactly as stored.
        dimension: Which dimension it belongs to, e.g. ``"city"``.
        column: The qualified column to filter on.
        confidence: How certain the match is, from 0 to 1.
        method: How it was matched, for the trace and for tests.
    """

    text: str
    value: str
    dimension: str
    column: str
    confidence: float
    method: str


class EntityCatalogue:
    """Every distinct dimension value in the database, indexed for lookup."""

    def __init__(self, values: dict[str, list[str]]) -> None:
        """Build the lookup indexes.

        Args:
            values: Dimension name to its distinct values.
        """
        self.values = values
        self._by_dimension = {
            dimension.name: dimension for dimension in DIMENSIONS
        }

        # Exact-value index, lowercased, mapping to every dimension that has
        # that value. "Delhi" is a city; nothing else collides in this data,
        # but the index does not assume that.
        self._exact: dict[str, list[tuple[str, str]]] = {}
        for dimension_name, dimension_values in values.items():
            for value in dimension_values:
                self._exact.setdefault(value.lower(), []).append(
                    (dimension_name, value)
                )

        # Identifier index keyed by prefix and integer, so ST7, st07 and ST007
        # all reach the same row.
        self._identifiers: dict[tuple[str, int], tuple[str, str]] = {}
        for dimension in DIMENSIONS:
            if not dimension.is_identifier:
                continue
            for value in values.get(dimension.name, []):
                parsed = _parse_identifier(value)
                if parsed is not None:
                    self._identifiers[parsed] = (dimension.name, value)

        # Alias index, filtered to aliases whose target actually exists.
        self._aliases: dict[str, list[tuple[str, str]]] = {}
        for alias, canonical in CITY_ALIASES.items():
            if canonical.lower() in self._exact:
                self._aliases.setdefault(alias, []).extend(
                    entry
                    for entry in self._exact[canonical.lower()]
                    if entry[0] == "city"
                )
        for alias, canonical in CHANNEL_ALIASES.items():
            if canonical.lower() in self._exact:
                self._aliases.setdefault(alias, []).extend(
                    entry
                    for entry in self._exact[canonical.lower()]
                    if entry[0] == "channel"
                )
        delivery = [
            entry
            for channel in DELIVERY_CHANNELS
            for entry in self._exact.get(channel.lower(), [])
            if entry[0] == "channel"
        ]
        if delivery:
            self._aliases["delivery"] = delivery
            self._aliases["aggregator"] = delivery
            self._aliases["aggregators"] = delivery
            self._aliases["online"] = delivery

    def dimension(self, name: str) -> Dimension | None:
        """Look up a dimension by name.

        Args:
            name: The dimension name.

        Returns:
            The dimension, or None when it is not one of ours.
        """
        return self._by_dimension.get(name)

    def exact(self, text: str) -> list[tuple[str, str]]:
        """Find dimension values equal to this text, ignoring case.

        Args:
            text: The candidate phrase.

        Returns:
            (dimension name, canonical value) pairs.
        """
        return list(self._exact.get(text.lower(), []))

    def alias(self, text: str) -> list[tuple[str, str]]:
        """Find dimension values this text is a known alternate name for.

        Args:
            text: The candidate phrase.

        Returns:
            (dimension name, canonical value) pairs.
        """
        return list(self._aliases.get(text.lower(), []))

    def identifier(self, text: str) -> tuple[str, str] | None:
        """Find the entity an identifier refers to, ignoring zero padding.

        Args:
            text: The candidate token, e.g. ``"ST7"``.

        Returns:
            (dimension name, canonical value), or None when it matches nothing.
        """
        parsed = _parse_identifier(text)
        if parsed is None:
            return None
        return self._identifiers.get(parsed)

    def names_for(self, dimension_name: str) -> list[str]:
        """Every distinct value of one dimension.

        Args:
            dimension_name: The dimension.

        Returns:
            Its values, or an empty list when unknown.
        """
        return list(self.values.get(dimension_name, []))


def _parse_identifier(text: str) -> tuple[str, int] | None:
    """Split an identifier into its prefix and numeric part.

    This is what lets ``ST7``, ``st07`` and ``ST007`` all resolve to the same
    store: the padding is discarded and the integer compared.

    Args:
        text: The candidate token.

    Returns:
        (lowercase prefix, integer), or None when the token is not an
        identifier.
    """
    match = _IDENTIFIER.match(text.strip())
    if match is None:
        return None
    return match.group("prefix").lower(), int(match.group("digits"))


def _load_values(db_path: str) -> dict[str, list[str]]:
    """Read every distinct dimension value from the database.

    Args:
        db_path: Path to the SQLite file.

    Returns:
        Dimension name to sorted distinct values. A dimension whose table is
        missing yields an empty list rather than raising: entity resolution is
        an enhancement, and an incomplete catalogue must not stop the system
        answering questions.
    """
    values: dict[str, list[str]] = {}
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as error:
        logger.warning("entity_catalogue_unavailable", extra={"error": str(error)})
        return {dimension.name: [] for dimension in DIMENSIONS}

    try:
        for dimension in DIMENSIONS:
            column = dimension.column.split(".")[-1]
            try:
                rows = connection.execute(
                    f"SELECT DISTINCT {column} FROM {dimension.table} "  # noqa: S608
                    f"WHERE {column} IS NOT NULL AND TRIM({column}) <> ''"
                ).fetchall()
            except sqlite3.Error as error:
                logger.warning(
                    "entity_dimension_unavailable",
                    extra={"dimension": dimension.name, "error": str(error)},
                )
                values[dimension.name] = []
                continue
            values[dimension.name] = sorted(str(row[0]) for row in rows)
    finally:
        connection.close()

    logger.info(
        "entity_catalogue_loaded",
        extra={
            "dimensions": len(values),
            "values": sum(len(entries) for entries in values.values()),
        },
    )
    return values


@lru_cache(maxsize=1)
def get_catalogue(db_path: str | None = None) -> EntityCatalogue:
    """Return the shared catalogue, loading it on first use.

    Args:
        db_path: Database to read. Defaults to the configured path.

    Returns:
        The cached catalogue.
    """
    return EntityCatalogue(_load_values(db_path or str(settings.DB_PATH)))


def reset_catalogue() -> None:
    """Discard the cached catalogue so the next call reloads it.

    Used by tests that build a database after the catalogue was first read.
    """
    get_catalogue.cache_clear()


def _phrases(question: str, max_words: int = 4) -> list[tuple[str, int, int]]:
    """Enumerate candidate phrases from a question, longest first.

    Multi-word phrases are needed because "Veg Burger 5" and "New Year" are
    single values that no single token would find. Longest-first ordering means
    a specific match consumes its words before a shorter one can claim them.

    Args:
        question: The user's question.
        max_words: Longest phrase to consider.

    Returns:
        (phrase, start token index, end token index) tuples, ordered by
        decreasing length.
    """
    tokens = [
        (match.group(0), match.start(), match.end())
        for match in _TOKEN.finditer(question)
    ]
    candidates: list[tuple[str, int, int]] = []
    for size in range(min(max_words, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            window = tokens[start : start + size]
            phrase = question[window[0][1] : window[-1][2]]
            candidates.append((phrase, start, start + size - 1))
    return candidates


def _singular(text: str) -> str:
    """Strip a trailing plural 's' so "burgers" can match "Burger".

    Args:
        text: The candidate phrase.

    Returns:
        The phrase without a trailing 's', when removing one leaves something
        substantial.
    """
    lowered = text.lower()
    if len(lowered) > MIN_SUBSTRING_LENGTH and lowered.endswith("s"):
        return lowered[:-1]
    return lowered


def _contains_phrase(value: str, needle: str) -> bool:
    """Whether a value contains the needle starting at a word boundary.

    A plain substring test is wrong here in a way that produces a confidently
    incorrect answer: "veg burger" is a substring of "Non-Veg Burger 2", so
    asking about veg burgers would silently include the non-veg ones. Requiring
    the match to begin at the start of the value or after whitespace fixes
    exactly that, while still letting "burger" match "Non-Veg Burger 2" - which
    is correct, because that question really does include both.

    Args:
        value: The canonical value.
        needle: The lowercased phrase from the question.

    Returns:
        True when the value contains the needle at a word boundary.
    """
    return re.search(rf"(?:^|\s){re.escape(needle)}", value.lower()) is not None


def _substring_matches(
    phrase: str, catalogue: EntityCatalogue
) -> list[tuple[str, str]]:
    """Find dimension values containing this phrase as a word sequence.

    This is what resolves "veg burger" to every ``Veg Burger N`` SKU and
    "Gurugram" to the eight stores named after it.

    Args:
        phrase: The candidate phrase.
        catalogue: The loaded catalogue.

    Returns:
        (dimension name, canonical value) pairs.
    """
    needle = _singular(phrase)
    if len(needle) < MIN_SUBSTRING_LENGTH:
        return []

    found: list[tuple[str, str]] = []
    for dimension in DIMENSIONS:
        # Identifiers are matched numerically, never by substring: "ST01"
        # is a prefix of "ST010" and matching those together would be wrong.
        if dimension.is_identifier:
            continue
        for value in catalogue.names_for(dimension.name):
            if _contains_phrase(value, needle):
                found.append((dimension.name, value))
    return found


def _fuzzy_matches(
    phrase: str, catalogue: EntityCatalogue
) -> list[tuple[str, str, float]]:
    """Find dimension values within the typo threshold of this phrase.

    Fuzzy matching is restricted to single tokens. A typo spanning several
    words is rare, while multi-word fuzzy matching is where the false positives
    come from: "in Bengaluru" scores 0.86 against "Bengaluru" and would claim
    those words before the bare token could match it exactly.

    Args:
        phrase: The candidate phrase.
        catalogue: The loaded catalogue.

    Returns:
        (dimension name, canonical value, ratio) triples above the threshold.
    """
    lowered = phrase.lower().strip()
    if len(lowered) < MIN_FUZZY_LENGTH or len(_TOKEN.findall(lowered)) > 1:
        return []

    found: list[tuple[str, str, float]] = []
    for dimension in DIMENSIONS:
        if dimension.is_identifier:
            continue
        for value in catalogue.names_for(dimension.name):
            ratio = difflib.SequenceMatcher(None, lowered, value.lower()).ratio()
            if ratio >= FUZZY_THRESHOLD:
                found.append((dimension.name, value, ratio))
    return found


def resolve(
    text: str, catalogue: EntityCatalogue | None = None
) -> list[EntityMatch]:
    """Resolve every entity a question names to values in the data.

    Matching is tried in decreasing order of certainty - exact, alias,
    identifier, substring, then fuzzy - and the first kind that produces
    anything for a phrase wins, so a store named exactly is never also offered
    as a fuzzy near-miss of a different store. Words already consumed by a
    longer phrase are not reconsidered, which stops "Veg Burger 5" also
    resolving as "Veg".

    Args:
        text: The user's question.
        catalogue: Catalogue to resolve against. Defaults to the shared one.

    Returns:
        Matches ordered by confidence then dimension, capped at
        :data:`MAX_MATCHES`. Empty when the question names nothing in the data,
        which is the correct answer for a question about competitors or for one
        naming a store that does not exist.
    """
    catalogue = catalogue or get_catalogue()
    matches: list[EntityMatch] = []
    consumed: set[int] = set()

    for phrase, start, end in _phrases(text):
        span = set(range(start, end + 1))
        if span & consumed:
            continue
        lowered = phrase.lower().strip()
        if not lowered or (lowered in STOPWORDS and start == end):
            continue

        found = _match_phrase(phrase, catalogue)
        if not found:
            continue
        matches.extend(found)
        consumed |= span

    matches.sort(key=lambda match: (-match.confidence, match.dimension, match.value))
    return matches[:MAX_MATCHES]


def _match_phrase(
    phrase: str, catalogue: EntityCatalogue
) -> list[EntityMatch]:
    """Resolve one phrase, stopping at the most certain kind that matches.

    Args:
        phrase: The candidate phrase.
        catalogue: The loaded catalogue.

    Returns:
        Matches for this phrase, empty when it names nothing.
    """
    stripped = phrase.strip()

    exact = catalogue.exact(stripped)
    if exact:
        return [
            _build(stripped, dimension, value, catalogue, _exact_confidence(
                stripped, value
            ), "exact")
            for dimension, value in exact
        ]

    alias = catalogue.alias(stripped)
    if alias:
        return [
            _build(stripped, dimension, value, catalogue, CONFIDENCE_ALIAS, "alias")
            for dimension, value in alias
        ]

    identifier = catalogue.identifier(stripped)
    if identifier is not None:
        dimension, value = identifier
        confidence = (
            CONFIDENCE_EXACT
            if stripped.upper() == value.upper()
            else CONFIDENCE_NORMALISED_ID
        )
        return [
            _build(stripped, dimension, value, catalogue, confidence, "identifier")
        ]

    substring = _substring_matches(stripped, catalogue)
    if substring:
        return [
            _build(
                stripped, dimension, value, catalogue, CONFIDENCE_SUBSTRING,
                "substring",
            )
            for dimension, value in substring
        ]

    fuzzy = _fuzzy_matches(stripped, catalogue)
    if fuzzy:
        best = max(ratio for _, _, ratio in fuzzy)
        return [
            _build(
                stripped,
                dimension,
                value,
                catalogue,
                round(CONFIDENCE_FUZZY_BASE * ratio, 3),
                "fuzzy",
            )
            for dimension, value, ratio in fuzzy
            # Only the closest fuzzy candidates; a long tail of near-misses is
            # noise that would make everything look ambiguous.
            if ratio >= best - 0.02
        ]

    return []


def _exact_confidence(phrase: str, value: str) -> float:
    """Score an exact match, distinguishing identical from case-folded.

    Args:
        phrase: The text from the question.
        value: The canonical value.

    Returns:
        The confidence.
    """
    return CONFIDENCE_EXACT if phrase == value else CONFIDENCE_CASE_INSENSITIVE


def _build(
    text: str,
    dimension_name: str,
    value: str,
    catalogue: EntityCatalogue,
    confidence: float,
    method: str,
) -> EntityMatch:
    """Assemble one match, attaching the column to filter on.

    Args:
        text: The matched text from the question.
        dimension_name: Which dimension matched.
        value: The canonical value.
        catalogue: The loaded catalogue.
        confidence: Match confidence.
        method: How it matched.

    Returns:
        The match.
    """
    dimension = catalogue.dimension(dimension_name)
    return EntityMatch(
        text=text,
        value=value,
        dimension=dimension_name,
        column=dimension.column if dimension else dimension_name,
        confidence=confidence,
        method=method,
    )


def group_by_dimension(
    matches: Iterable[EntityMatch],
) -> dict[str, list[EntityMatch]]:
    """Group matches by the dimension they belong to.

    Args:
        matches: The matches to group.

    Returns:
        Dimension name to its matches, preserving input order within a group.
    """
    grouped: dict[str, list[EntityMatch]] = {}
    for match in matches:
        grouped.setdefault(match.dimension, []).append(match)
    return grouped


def ambiguous_references(
    matches: Iterable[EntityMatch],
) -> dict[str, list[EntityMatch]]:
    """Find question phrases that matched several entities equally well.

    A phrase resolving to several values of the *same* dimension is usually
    fine - "the Gurugram stores" legitimately means eight stores, and a plan
    can cover them all. A phrase resolving across *different* dimensions with
    similar confidence is the ambiguity worth surfacing: "Gurugram" could mean
    the city or a store named after it, and those give different answers.

    Args:
        matches: The resolved matches.

    Returns:
        The matched text to its competing candidates, for phrases that are
        genuinely ambiguous. Empty when nothing is.
    """
    by_text: dict[str, list[EntityMatch]] = {}
    for match in matches:
        by_text.setdefault(match.text.lower(), []).append(match)

    ambiguous: dict[str, list[EntityMatch]] = {}
    for text, candidates in by_text.items():
        dimensions = {match.dimension for match in candidates}
        if len(dimensions) < 2:
            continue
        best = max(match.confidence for match in candidates)
        close = [
            match
            for match in candidates
            if best - match.confidence <= AMBIGUITY_BAND
        ]
        if len({match.dimension for match in close}) >= 2:
            ambiguous[text] = close
    return ambiguous


def render_for_prompt(matches: Iterable[EntityMatch]) -> str:
    """Render resolved entities as an instruction block for the planner.

    The block states the canonical values and the columns they live in, so the
    planner filters on what exists rather than on what it would have guessed
    from the wording of the question.

    Args:
        matches: The resolved matches.

    Returns:
        The prompt block, or an empty string when nothing resolved - in which
        case no block is added at all rather than one saying "none", which
        would invite the model to explain its absence.
    """
    matches = list(matches)
    if not matches:
        return ""

    grouped = group_by_dimension(matches)
    lines = [
        "RESOLVED ENTITIES",
        "",
        "The question refers to these values, already matched against the "
        "database. Filter on EXACTLY these canonical values and columns. Do "
        "not re-spell them, do not guess variants, and do not filter on the "
        "user's wording where it differs from the canonical value.",
        "",
    ]
    for dimension_name, group in grouped.items():
        column = group[0].column
        rendered = ", ".join(
            f"'{match.value}'"
            + (f" (from \"{match.text}\")" if match.text.lower() != match.value.lower() else "")
            for match in group[:12]
        )
        more = "" if len(group) <= 12 else f", and {len(group) - 12} more"
        lines.append(f"- {dimension_name} -> filter {column} IN ({rendered}{more})")

    ambiguous = ambiguous_references(matches)
    if ambiguous:
        lines.append("")
        lines.append(
            "AMBIGUOUS: these phrases matched more than one kind of entity. "
            "Either cover every candidate in the plan, or set "
            "intent=\"ambiguous\" and ask which was meant."
        )
        for text, candidates in ambiguous.items():
            options = "; ".join(
                f"{match.dimension} '{match.value}'" for match in candidates[:6]
            )
            lines.append(f'- "{text}" -> {options}')

    return "\n".join(lines)
