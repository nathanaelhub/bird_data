"""Tests for the ETL transformations in etl.py.

These run against tiny hand-built fixtures (tests/fixtures/), so they need no
network and no downloaded source data. Each test pins one behaviour the manual
spreadsheet workflow used to get wrong: date typing, dropping incomplete rows,
filtering to migration months, label normalisation, the derived season/decade
features, and the date-based light-score join.
"""
from pathlib import Path

import pandas as pd
import pytest

from birddata import etl

FIXTURES = Path(__file__).parent / "fixtures"
COLLISIONS = FIXTURES / "bird_collisions_sample.csv"
LIGHT = FIXTURES / "mp_light_sample.csv"


@pytest.fixture
def clean():
    return etl.clean_collisions(COLLISIONS)


def test_drops_incomplete_and_offseason_rows(clean):
    # fixture has 9 rows; dropped: missing date, missing species, and a July
    # (off-migration-month) record -> 6 survive.
    assert len(clean) == 6


def test_no_dates_survive_as_null(clean):
    assert clean["date"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(clean["date"])


def test_only_migration_months_remain(clean):
    assert set(clean["month"]).issubset(set(etl.SEASON))
    assert 7 not in set(clean["month"])  # the July record was filtered out


def test_labels_are_normalised(clean):
    # " Passerellidae " -> "Passerellidae"; no surrounding whitespace anywhere
    assert (clean["family"] == clean["family"].str.strip()).all()
    # "yes"/"Yes" both normalise to title case
    assert set(clean["flight_call"]).issubset({"Yes", "No"})


def test_locality_is_expanded(clean):
    assert set(clean["locality"]).issubset({"McCormick Place", "Greater Chicago"})


def test_scientific_name_and_features(clean):
    row = clean[clean["date"] == "1978-10-27"].iloc[0]
    assert row["scientific_name"] == "Passerculus sandwichensis"
    assert row["season"] == "Fall"
    assert row["decade"] == 1970
    spring = clean[clean["date"] == "2005-05-10"].iloc[0]
    assert spring["season"] == "Spring"


def test_build_mp_daily_join_and_grouping(clean):
    daily = etl.build_mp_daily(clean, LIGHT)
    # MP nights with a matching light score: 1978-10-27, 2005-05-10, 2012-04-02.
    # 1979-10-23 (MP, no light row) and the CHI record are excluded.
    assert len(daily) == 3

    by_date = daily.set_index(daily["date"].dt.strftime("%Y-%m-%d"))["collisions"]
    assert by_date["2005-05-10"] == 2  # two same-day collisions collapse to one night
    assert by_date["1978-10-27"] == 1
    assert "1979-10-23" not in by_date.index  # no light score -> dropped by inner join

    light = daily.set_index(daily["date"].dt.strftime("%Y-%m-%d"))["light_score"]
    assert light["2012-04-02"] == 5
