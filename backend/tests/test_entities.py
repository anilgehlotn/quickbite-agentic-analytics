"""Tests for entity resolution.

Two properties are load-bearing and get the most attention here:

* **Never invent.** ``TestNeverInvents`` asserts that a store id, a city and a
  product that do not exist resolve to nothing. A resolver that helpfully
  substitutes the nearest real value turns "how is ST999 doing" into a
  confident answer about a different store, which is the worst failure this
  module could have.
* **Never widen.** ``test_veg_burgers_exclude_non_veg`` guards a real bug: "veg
  burger" is a plain substring of "Non-Veg Burger 2", so a naive contains-test
  answers a question about veg burgers with non-veg data.

The catalogue is read from the real database rather than a fixture, because the
whole point of the module is to match what is actually stored. A synthetic
catalogue would let a rename in the ETL pass unnoticed.
"""

from __future__ import annotations

import pytest

from app.semantic.entities import (
    CONFIDENCE_EXACT,
    DIMENSIONS,
    EntityCatalogue,
    EntityMatch,
    ambiguous_references,
    get_catalogue,
    group_by_dimension,
    render_for_prompt,
    resolve,
)


@pytest.fixture(scope="module")
def catalogue() -> EntityCatalogue:
    """Load the catalogue from the real database once for the module.

    Returns:
        The shared catalogue.
    """
    return get_catalogue()


def values_of(matches: list[EntityMatch], dimension: str) -> set[str]:
    """Collect the canonical values matched for one dimension.

    Args:
        matches: The resolver's output.
        dimension: The dimension to filter to.

    Returns:
        The distinct values.
    """
    return {match.value for match in matches if match.dimension == dimension}


class TestCatalogue:
    """The catalogue reflects what is actually in the database."""

    def test_every_dimension_is_loaded(self, catalogue: EntityCatalogue) -> None:
        """Each declared dimension has at least one value."""
        for dimension in DIMENSIONS:
            assert catalogue.names_for(dimension.name), dimension.name

    def test_known_counts(self, catalogue: EntityCatalogue) -> None:
        """Dimension sizes match the dataset's documented shape."""
        assert len(catalogue.names_for("store_id")) == 50
        assert len(catalogue.names_for("sku_id")) == 30
        assert len(catalogue.names_for("city")) == 8
        assert len(catalogue.names_for("channel")) == 4

    def test_columns_are_qualified(self) -> None:
        """Every dimension names a table-qualified column to filter on."""
        for dimension in DIMENSIONS:
            assert "." in dimension.column, dimension.name


class TestExactMatching:
    """Values written exactly as stored resolve with full confidence."""

    def test_store_id(self) -> None:
        """A canonical store id resolves exactly."""
        matches = resolve("How is ST007 doing?")
        assert values_of(matches, "store_id") == {"ST007"}
        assert matches[0].confidence == CONFIDENCE_EXACT
        assert matches[0].method == "exact"

    def test_column_is_attached(self) -> None:
        """The match carries the column the planner must filter on."""
        matches = resolve("How is ST007 doing?")
        assert matches[0].column == "dim_store.store_id"

    def test_case_insensitive(self) -> None:
        """Lowercase input still resolves, at slightly lower confidence."""
        matches = resolve("how is store st007 doing")
        assert values_of(matches, "store_id") == {"ST007"}
        assert matches[0].confidence < CONFIDENCE_EXACT

    def test_multi_word_value(self) -> None:
        """A two-word value resolves as one entity."""
        matches = resolve("How did New Year trading go?")
        assert values_of(matches, "festive_period") == {"New Year"}

    def test_hyphenated_value(self) -> None:
        """A hyphenated value resolves without being split."""
        matches = resolve("how are non-veg items doing")
        assert values_of(matches, "veg_nonveg") == {"Non-Veg"}

    def test_channel(self) -> None:
        """A channel name resolves to the channel dimension."""
        matches = resolve("How is Zomato performing?")
        assert values_of(matches, "channel") == {"Zomato"}


class TestIdentifierNormalisation:
    """Zero padding is not something a user should have to know."""

    @pytest.mark.parametrize("written", ["ST7", "st7", "ST07", "st007", "ST007"])
    def test_padding_variants_reach_the_same_store(self, written: str) -> None:
        """Every way of writing store 7 resolves to ST007."""
        matches = resolve(f"How is {written} doing?")
        assert values_of(matches, "store_id") == {"ST007"}

    def test_sku_padding(self) -> None:
        """SKU ids normalise the same way."""
        matches = resolve("How is SKU5 selling?")
        assert values_of(matches, "sku_id") == {"SKU005"}

    def test_padded_form_scores_below_exact(self) -> None:
        """An exactly-written id outranks one that needed normalising."""
        exact = resolve("ST007")[0]
        padded = resolve("ST7")[0]
        assert exact.confidence > padded.confidence

    def test_identifiers_do_not_prefix_match(self) -> None:
        """ST01 must not also resolve to ST010 and ST015."""
        matches = resolve("How is ST01 doing?")
        assert values_of(matches, "store_id") == {"ST001"}


class TestCityAliases:
    """People use the older names, and the data uses the current ones."""

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("Bangalore", "Bengaluru"),
            ("Bombay", "Mumbai"),
            ("Calcutta", "Kolkata"),
            ("Madras", "Chennai"),
            ("Gurgaon", "Gurugram"),
        ],
    )
    def test_alias_resolves_to_canonical(
        self, alias: str, canonical: str
    ) -> None:
        """Each historical name reaches the value in the data."""
        matches = resolve(f"What is revenue in {alias}?")
        assert canonical in values_of(matches, "city")

    def test_alias_only_offered_when_the_city_exists(
        self, catalogue: EntityCatalogue
    ) -> None:
        """An alias for a city not in this dataset resolves to nothing."""
        assert "Mysuru" not in catalogue.names_for("city")
        assert resolve("revenue in Mysore") == []

    def test_delivery_means_both_aggregators(self) -> None:
        """"Delivery" is not a channel, so it resolves to both that are."""
        matches = resolve("What share of revenue comes from delivery?")
        assert values_of(matches, "channel") == {"Swiggy", "Zomato"}


class TestFuzzyMatching:
    """Typos resolve; different words do not."""

    def test_misspelled_city(self) -> None:
        """A one-character slip still finds the city."""
        matches = resolve("revenue in Bengalru")
        assert values_of(matches, "city") == {"Bengaluru"}
        assert matches[0].method == "fuzzy"

    def test_fuzzy_confidence_is_lower_than_exact(self) -> None:
        """A guess is never reported as certain."""
        assert resolve("revenue in Bengalru")[0].confidence < CONFIDENCE_EXACT

    def test_short_tokens_are_not_fuzzy_matched(self) -> None:
        """Below the length floor, edit distance stops being evidence."""
        assert values_of(resolve("revenue in Pure"), "city") == set()

    def test_multi_word_phrases_are_not_fuzzy_matched(self) -> None:
        """A phrase must not claim words the exact token needs.

        "in Bengaluru" is within the fuzzy threshold of "Bengaluru"; allowing
        it would consume both tokens at 0.6 confidence and prevent the bare
        token matching exactly.
        """
        matches = resolve("best store in Bengaluru")
        assert values_of(matches, "city") == {"Bengaluru"}
        assert matches[0].method == "exact"


class TestPartialProductNames:
    """Product questions are rarely written as full SKU names."""

    def test_partial_name_matches_every_variant(self) -> None:
        """"veg burgers" reaches all three Veg Burger SKUs."""
        matches = resolve("how are veg burgers selling")
        assert values_of(matches, "sku_name") == {
            "Veg Burger 1",
            "Veg Burger 3",
            "Veg Burger 5",
        }

    def test_veg_burgers_exclude_non_veg(self) -> None:
        """The substring test must respect word boundaries.

        "veg burger" occurs inside "Non-Veg Burger 2". Matching on a plain
        substring would answer a question about veg burgers with non-veg data
        and nothing downstream could detect it.
        """
        matches = resolve("how are veg burgers selling")
        assert not any(
            value.startswith("Non-Veg") for value in values_of(matches, "sku_name")
        )

    def test_broader_term_includes_both(self) -> None:
        """"burger" legitimately covers veg and non-veg alike."""
        values = values_of(resolve("burger sales by month"), "sku_name")
        assert any(value.startswith("Veg Burger") for value in values)
        assert any(value.startswith("Non-Veg Burger") for value in values)

    def test_category_name_resolves_exactly_not_by_substring(self) -> None:
        """"Burgers" is the category's own name, so it wins outright.

        The exact tier stopping further matching is what keeps a question about
        a category from also dragging in every SKU in it.
        """
        matches = resolve("how are burgers selling")
        assert values_of(matches, "category") == {"Burgers"}
        assert values_of(matches, "sku_name") == set()

    def test_singular_and_plural_agree(self) -> None:
        """A trailing 's' does not change what resolves."""
        assert values_of(resolve("veg burger sales"), "sku_name") == values_of(
            resolve("veg burgers sales"), "sku_name"
        )


class TestNeverInvents:
    """A term matching nothing must resolve to nothing."""

    def test_unknown_store_id(self) -> None:
        """A store id outside the range resolves to nothing."""
        assert values_of(resolve("How is ST999 doing?"), "store_id") == set()

    def test_unknown_city(self) -> None:
        """A city not in the data does not become the nearest one."""
        assert values_of(resolve("revenue in Reykjavik"), "city") == set()

    def test_competitor_names(self) -> None:
        """A competitor resolves to no store or city."""
        matches = resolve("How do we compare with McDonalds?")
        assert values_of(matches, "store_name") == set()
        assert values_of(matches, "city") == set()

    def test_question_naming_nothing(self) -> None:
        """A purely analytical question resolves to nothing at all."""
        assert resolve("What was our total revenue in the last 3 months?") == []

    def test_analytical_vocabulary_is_not_an_entity(self) -> None:
        """Words like "best" and "worst" never resolve."""
        assert resolve("Which stores are performing best and worst?") == []


class TestAmbiguity:
    """Genuine ambiguity survives to the caller rather than being decided."""

    def test_term_matching_two_dimensions_is_ambiguous(self) -> None:
        """"pizzas" is both the category Pizza and several product names."""
        matches = resolve("how are pizzas selling")
        ambiguous = ambiguous_references(matches)
        assert ambiguous
        dimensions = {
            match.dimension
            for candidates in ambiguous.values()
            for match in candidates
        }
        assert {"category", "sku_name"} <= dimensions

    def test_ambiguity_spanning_three_dimensions(self) -> None:
        """"beverage" names a category, a promotion type and products."""
        ambiguous = ambiguous_references(resolve("beverage revenue"))
        dimensions = {
            match.dimension
            for candidates in ambiguous.values()
            for match in candidates
        }
        assert {"category", "promo_type", "sku_name"} <= dimensions

    def test_several_values_of_one_dimension_are_not_ambiguous(self) -> None:
        """Eight stores in one city is coverage, not ambiguity."""
        matches = resolve("how are veg burgers selling")
        assert ambiguous_references(matches) == {}

    def test_unambiguous_question_reports_none(self) -> None:
        """An exact identifier is never ambiguous."""
        assert ambiguous_references(resolve("How is ST007 doing?")) == {}


class TestCombinations:
    """Real questions name several things at once."""

    def test_city_channel_and_product(self) -> None:
        """Each reference resolves to its own dimension."""
        matches = resolve("revenue for Mumbai stores on Zomato")
        assert values_of(matches, "city") == {"Mumbai"}
        assert values_of(matches, "channel") == {"Zomato"}

    def test_two_stores_compared(self) -> None:
        """Both sides of a comparison resolve."""
        matches = resolve("compare ST7 and ST15")
        assert values_of(matches, "store_id") == {"ST007", "ST015"}

    def test_segments(self) -> None:
        """Customer segments resolve from ordinary wording."""
        matches = resolve("How do loyal customers compare with occasional ones?")
        assert values_of(matches, "customer_segment") == {"Loyal", "Occasional"}

    def test_store_format(self) -> None:
        """A two-word format resolves as one value."""
        assert values_of(resolve("how do food court stores do"), "store_format") == {
            "Food Court"
        }

    def test_grouping(self) -> None:
        """Matches group by the dimension they belong to."""
        grouped = group_by_dimension(resolve("Mumbai on Zomato"))
        assert set(grouped) == {"city", "channel"}


class TestPromptRendering:
    """The prompt block states canonical values and columns."""

    def test_empty_when_nothing_resolved(self) -> None:
        """No entities means no block, not a block saying "none"."""
        assert render_for_prompt([]) == ""

    def test_names_the_column_and_value(self) -> None:
        """The block tells the planner exactly what to filter on."""
        rendered = render_for_prompt(resolve("How is ST007 doing?"))
        assert "dim_store.store_id" in rendered
        assert "'ST007'" in rendered

    def test_shows_the_users_wording_when_it_differs(self) -> None:
        """A normalised id shows where it came from."""
        rendered = render_for_prompt(resolve("How is ST7 doing?"))
        assert "ST007" in rendered
        assert "ST7" in rendered

    def test_flags_ambiguity(self) -> None:
        """An ambiguous reference is called out for the planner."""
        rendered = render_for_prompt(resolve("how are pizzas selling"))
        assert "AMBIGUOUS" in rendered

    def test_caps_the_listed_values(self) -> None:
        """A question naming a whole dimension cannot flood the prompt."""
        rendered = render_for_prompt(resolve("beverage revenue"))
        assert len(rendered) < 4000
