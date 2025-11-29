import numpy as np
import pandas as pd

def auto_clean(df, drop_all_null_cols=True, drop_duplicates=True):
    """
    Automatically clean common issues in a pandas DataFrame.
    Returns a cleaned DataFrame.
    """

    df = df.copy()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    for col in df.select_dtypes(include=['object']).columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid = parsed.notna().sum()
            if valid >= max(1, len(parsed) // 2):
                df[col] = parsed
        except Exception:
            pass

    if drop_all_null_cols:
        df.dropna(axis=1, how="all", inplace=True)

    if drop_duplicates:
        df.drop_duplicates(inplace=True)

    return df
