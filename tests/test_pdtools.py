import numpy as np
import pandas as pd
import pytest

from datatoolbelt import pdtools


@pytest.mark.parametrize("datatype", [list, tuple, np.array, pd.Series])
def test_freq(datatype):
    values = datatype(["a", "c", "b", "g", "h", "a", "g", "a"])
    actual = pdtools.freq(values)
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
