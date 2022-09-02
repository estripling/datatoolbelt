import numpy as np
import pandas as pd
from bumbag.core import flatten


def entropy(values):
    """Compute the Shannon entropy for discrete values.

    The Shannon entropy of a discrete random variable :math:`X` with support
    :math:`\\mathcal{X} \\subseteq \\Omega` and probability mass function
    :math:`p(x) = \\Pr(X = x) \\in (0, 1]` is

    .. math::

        H(X) \\triangleq \\mathbb{E}[ -\\log_{2} p(X) ]
        = - \\sum_{x \\in \\mathcal{X}} p(x) \\log_{2} p(x) \\in [0, \\infty).

    Parameters
    ----------
    values : array-like
        An input array for which entropy is to be computed.
        It must be 1-dimensional.

    Returns
    -------
    float
        Shannon entropy with log to the base 2.
        Returns NaN if input array is empty.

    References
    ----------
    .. [1] C. E. Shannon, "A Mathematical Theory of Communication,"
           in The Bell System Technical Journal, vol. 27, no. 3, pp. 379-423,
           July 1948, doi: 10.1002/j.1538-7305.1948.tb01338.x.

    Examples
    --------
    >>> entropy([])
    nan

    >>> entropy(["a", "a"])
    0.0

    >>> entropy(["a", "b"])
    1.0

    >>> entropy(["a", "b", "c", "d"])
    2.0
    """
    if len(values) == 0:
        return float("nan")

    counts = (
        pd.Series(values)
        .value_counts(normalize=False, sort=False, dropna=False)
        .values
    )

    if len(counts) == 1:
        return 0.0

    total = counts.sum()

    return float(
        np.log2(total)
        if len(counts) == total
        else np.log2(total) - (counts * np.log2(counts)).sum() / total
    )


def freq(values):
    """Compute value frequencies.

    Given an input array, calculate for each distinct value:
     - the frequency (``n``),
     - the cumulative frequency (``N``),
     - the relative frequency (``r``), and
     - the cumulative relative frequency (``R``).

    Parameters
    ----------
    values : array-like
        An input array of values to compute the frequencies of its members.
        It must be 1-dimensional.

    Returns
    -------
    pandas.DataFrame
        Frequencies of distinct values.

    Examples
    --------
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


def join_dataframes_by_index(*dataframes):
    """Join multiple dataframes by their index.

    Parameters
    ----------
    dataframes : sequence of pandas.DataFrame and pandas.Series
        Dataframes to join. Being a variadic function, it can handle in one
        call both cases when dataframes are given individually and when they
        are given in a sequence. The function also accepts pandas series.
        If an object is of type pandas.Series, it is converted to a dataframe.

    Returns
    -------
    pandas.DataFrame
        A new dataframe with all columns.

    Examples
    --------
    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["c", "d"])
    >>> df = join_dataframes_by_index(df1, df2)
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
       a  b  c  d
    0  1  2  5  6
    1  3  4  7  8

    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=[0, 1], columns=["a", "b"])
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], index=[0, 2], columns=["c", "d"])
    >>> df = join_dataframes_by_index([df1, df2])
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
         a    b    c    d
    0  1.0  2.0  5.0  6.0
    1  3.0  4.0  NaN  NaN
    2  NaN  NaN  7.0  8.0

    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    >>> s1 = pd.Series([5, 6], name="c")
    >>> df = join_dataframes_by_index(df1, s1)
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
       a  b  c
    0  1  2  5
    1  3  4  6

    >>> s1 = pd.Series([1, 2])
    >>> s2 = pd.Series([3, 4])
    >>> s3 = pd.Series([5, 6])
    >>> df = join_dataframes_by_index([s1, s2], s3)
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
       0  0  0
    0  1  3  5
    1  2  4  6

    >>> s1 = pd.Series([1, 2], index=[0, 1], name="a")
    >>> s2 = pd.Series([3, 4], index=[1, 2], name="b")
    >>> s3 = pd.Series([5, 6], index=[2, 3], name="c")
    >>> df = join_dataframes_by_index(s1, s2, s3)
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
         a    b    c
    0  1.0  NaN  NaN
    1  2.0  3.0  NaN
    2  NaN  4.0  5.0
    3  NaN  NaN  6.0
    """
    return pd.concat(map(pd.DataFrame, flatten(dataframes)), axis=1)


def union_dataframes_by_name(*dataframes):
    """Union multiple dataframes by their name.

    Parameters
    ----------
    dataframes : sequence of pandas.DataFrame and pandas.Series
        Dataframes to union. Being a variadic function, it can handle in one
        call both cases when dataframes are given individually and when they
        are given in a sequence. The function also accepts pandas series.
        If an object is of type pandas.Series, it is converted to a dataframe.

    Returns
    -------
    pandas.DataFrame
        A new dataframe with all rows.

    Notes
    -----
    - Inspired by the unionByName method of spark dataframe.
    - Deduplication is not performed on the returned dataframe.

    Examples
    --------
    >>> df1 = pd.DataFrame([[1, 2], [3, 4]])
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]])
    >>> df = union_dataframes_by_name(df1, df2)
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
       0  1
    0  1  2
    1  3  4
    0  5  6
    1  7  8

    >>> df1 = pd.DataFrame([[1, 1], [1, 1]])
    >>> df2 = pd.DataFrame([[1, 1], [1, 1]])
    >>> df = union_dataframes_by_name(df1, df2)
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
       0  1
    0  1  1
    1  1  1
    0  1  1
    1  1  1

    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=[0, 1])
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], index=[0, 2])
    >>> df = union_dataframes_by_name([df1, df2])
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
       0  1
    0  1  2
    1  3  4
    0  5  6
    2  7  8

    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=[0, 1], columns=["a", "b"])
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], index=[0, 2], columns=["c", "d"])
    >>> df = union_dataframes_by_name([df1, df2])
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
         a    b    c    d
    0  1.0  2.0  NaN  NaN
    1  3.0  4.0  NaN  NaN
    0  NaN  NaN  5.0  6.0
    2  NaN  NaN  7.0  8.0

    >>> df1 = pd.DataFrame([[1, 2], [3, 4]])
    >>> s1 = pd.Series([5, 6])
    >>> df = union_dataframes_by_name(df1, s1)
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
       0    1
    0  1  2.0
    1  3  4.0
    0  5  NaN
    1  6  NaN

    >>> s1 = pd.Series([1, 2])
    >>> s2 = pd.Series([3, 4])
    >>> s3 = pd.Series([5, 6])
    >>> df = union_dataframes_by_name([s1, s2], s3)
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
       0
    0  1
    1  2
    0  3
    1  4
    0  5
    1  6

    >>> s1 = pd.Series([1, 2], index=[0, 1], name="a")
    >>> s2 = pd.Series([3, 4], index=[1, 2], name="b")
    >>> s3 = pd.Series([5, 6], index=[2, 3], name="c")
    >>> df = union_dataframes_by_name(s1, s2, s3)
    >>> isinstance(df, pd.DataFrame)
    True
    >>> df
         a    b    c
    0  1.0  NaN  NaN
    1  2.0  NaN  NaN
    1  NaN  3.0  NaN
    2  NaN  4.0  NaN
    2  NaN  NaN  5.0
    3  NaN  NaN  6.0
    """
    return pd.concat(map(pd.DataFrame, flatten(dataframes)), axis=0)
