import numpy as np
import pandas as pd
import pytest

from datatoolbelt import pandastools


@pytest.mark.parametrize(
    "values",
    [
        [],
        ["a"],
        ["a", "a"],
        ["a", "b"],
        ["a", "b", "c", "d"],
        ["a", "a", "a", "a"],
        ["a", "a", "b", "b"],
        ["a", "b", "b", "c", "c", "c"],
        ["a", "a", "b", "b", "a", "a", "a"],
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        [1],
        [1.0] * 4,
        [1, 1, 2],
        [1 / 3, 1 / 3, 1 / 3],
        [np.pi, np.e, np.sqrt(2), np.pi],
        [1, 2, 3, 4, 5],
        range(10),
        [1, 2, np.nan, 4, 5, np.nan],
        [None] * 4,
        [float("nan")] * 4,
        [np.nan] * 4,
        [True],
        [True] * 4,
        [True, False],
        [True, False] * 2,
        [True, False, True, False, True],
    ],
)
@pytest.mark.parametrize("dropna", [True, False])
def test_efficiency(values, dropna):
    actual = pandastools.efficiency(values, dropna)
    assert isinstance(actual, float)

    if len(values) == 0:
        assert np.isnan(actual)
    else:
        probs = (
            pd.Series(values)
            .value_counts(normalize=True, sort=False, dropna=dropna)
            .values
        )
        expected = (
            float("nan")
            if len(probs) == 0
            else 0.0
            if len(probs) == 1
            else -(probs * np.log2(probs)).sum() / np.log2(len(probs))
        )
        np.testing.assert_almost_equal(actual, expected)


@pytest.mark.parametrize(
    "values",
    [
        [],
        ["a"],
        ["a", "a"],
        ["a", "b"],
        ["a", "b", "c", "d"],
        ["a", "a", "a", "a"],
        ["a", "a", "b", "b"],
        ["a", "b", "b", "c", "c", "c"],
        ["a", "a", "b", "b", "a", "a", "a"],
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        [1],
        [1.0] * 4,
        [1, 1, 2],
        [1 / 3, 1 / 3, 1 / 3],
        [np.pi, np.e, np.sqrt(2), np.pi],
        [1, 2, 3, 4, 5],
        range(10),
        [1, 2, np.nan, 4, 5, np.nan],
        [None] * 4,
        [float("nan")] * 4,
        [np.nan] * 4,
        [True],
        [True] * 4,
        [True, False],
        [True, False] * 2,
        [True, False, True, False, True],
    ],
)
@pytest.mark.parametrize("dropna", [True, False])
def test_entropy(values, dropna):
    actual = pandastools.entropy(values, dropna)
    assert isinstance(actual, float)

    if len(values) == 0:
        assert np.isnan(actual)
    else:
        probs = (
            pd.Series(values)
            .value_counts(normalize=True, sort=False, dropna=dropna)
            .values
        )
        expected = (
            float("nan")
            if len(probs) == 0
            else 0.0
            if len(probs) < 2
            else -(probs * np.log2(probs)).sum()
        )
        np.testing.assert_almost_equal(actual, expected)


@pytest.mark.parametrize("datatype", [list, tuple, np.array, pd.Series])
def test_freq(datatype):
    values = datatype(["a", "c", "b", "g", "h", "a", "g", "a"])
    actual = pandastools.freq(values)
    assert isinstance(actual, pd.DataFrame)

    expected = pd.DataFrame(
        [
            dict(n=3, N=3, r=0.375, R=0.375),
            dict(n=2, N=5, r=0.250, R=0.625),
            dict(n=1, N=6, r=0.125, R=0.750),
            dict(n=1, N=7, r=0.125, R=0.875),
            dict(n=1, N=8, r=0.125, R=1.000),
        ],
        index=["a", "g", "c", "b", "h"],
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_join_dataframes_by_index():
    idx = (0, 1)

    cols1 = ("a", "b")
    row11, row12 = (1, 2), (3, 4)
    df1 = pd.DataFrame([row11, row12], idx, cols1)

    cols2 = ("c", "d")
    row21, row22 = (5, 6), (7, 8)
    df2 = pd.DataFrame([row21, row22], idx, cols2)

    expected = pd.DataFrame([row11 + row21, row12 + row22], idx, cols1 + cols2)

    actual = pandastools.join_dataframes_by_index(df1, df2)
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected)


def test_union_dataframes_by_name():
    cols = ("a", "b")

    idx1 = (0, 1)
    row11, row12 = (1, 2), (3, 4)
    df1 = pd.DataFrame([row11, row12], idx1, cols)

    idx2 = (2, 3)
    row21, row22 = (5, 6), (7, 8)
    df2 = pd.DataFrame([row21, row22], idx2, cols)

    expected = pd.DataFrame([row11, row12, row21, row22], idx1 + idx2, cols)

    actual = pandastools.union_dataframes_by_name(df1, df2)
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected)
