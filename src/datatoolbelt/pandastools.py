import pandas as pd
from bumbag.core import flatten


def freq(values):
    """Compute value frequencies.

    Given a collection of values, calculate for each value:
     - the frequency (``n``),
     - the cumulative frequency (``N``),
     - the relative frequency (``r``), and
     - the cumulative relative frequency (``R``).

    Parameters
    ----------
    values : array_like
        Collection of values to evaluate.

    Returns
    -------
    pandas.DataFrame
        Frequencies of each distinct value.

    Examples
    --------
    >>> import pandas as pd
    >>> x = ["a", "c", "b", "g", "h", "a", "g", "a"]
    >>> frequency = freq(x)
    >>> isinstance(frequency, pd.DataFrame)
    True
    >>> frequency
       n  N      r      R
    a  3  3  0.375  0.375
    g  2  5  0.250  0.625
    c  1  6  0.125  0.750
    b  1  7  0.125  0.875
    h  1  8  0.125  1.000
    """
    return pd.DataFrame(
        data=pd.Series(values).value_counts(
            sort=True,
            ascending=False,
            bins=None,
            dropna=False,
        ),
        columns=["n"],
    ).assign(
        N=lambda df: df["n"].cumsum(),
        r=lambda df: df["n"] / df["n"].sum(),
        R=lambda df: df["r"].cumsum(),
    )


def join_pandas_dataframes_by_index(*dataframes):
    """Join multiple dataframes by their index.

    Parameters
    ----------
    dataframes : pandas.DataFrame or collection of pandas.DataFrame
        Collection of dataframes to join.

    Returns
    -------
    pandas.DataFrame
        Joined dataframe.

    Notes
    -----
    - It is a variadic function.
    - It accepts a collection of dataframes and/or indvidual dataframes.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["c", "d"])
    >>> df = join_pandas_dataframes_by_index(df1, df2)
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
       a  b  c  d
    0  1  2  5  6
    1  3  4  7  8

    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=[0, 1], columns=["a", "b"])
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], index=[0, 2], columns=["c", "d"])
    >>> df = join_pandas_dataframes_by_index([df1, df2])
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
         a    b    c    d
    0  1.0  2.0  5.0  6.0
    1  3.0  4.0  NaN  NaN
    2  NaN  NaN  7.0  8.0
    """
    return pd.concat(flatten(dataframes), axis=1)
