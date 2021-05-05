from pandas.util.testing import assert_frame_equal
from pandas.util.testing import assert_series_equal


def assertFrameEqual(df1, df2, **kwds):
    """ Assert that two dataframes are equal, ignoring ordering of columns"""
    if df1.empty and df2.empty:
        df2 = df1
    return assert_frame_equal(df1.sort_index(axis=1).reset_index(drop=True),
                              df2.sort_index(axis=1).reset_index(drop=True),
                              check_names=True, **kwds)


def assertSeriesEqual(df1, df2, **kwds):
    """ Assert that two series are equal, ignoring ordering of columns"""
    if df1.empty and df2.empty:
        df2 = df1
    return assert_series_equal(df1.sort_index().reset_index(drop=True),
                               df2.sort_index().reset_index(drop=True),
                               check_names=False, **kwds)

