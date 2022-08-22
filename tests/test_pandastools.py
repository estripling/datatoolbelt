import numpy as np
import pandas as pd
import pytest

from datatoolbelt import pandastools


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


def test_join_pandas_dataframes_by_index():
    idx = (0, 1)

    cols1 = ("a", "b")
    row11, row12 = (1, 2), (3, 4)
    df1 = pd.DataFrame([row11, row12], idx, cols1)

    cols2 = ("c", "d")
    row21, row22 = (5, 6), (7, 8)
    df2 = pd.DataFrame([row21, row22], idx, cols2)

    expected = pd.DataFrame([row11 + row21, row12 + row22], idx, cols1 + cols2)

    actual = pandastools.join_pandas_dataframes_by_index(df1, df2)
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected)
