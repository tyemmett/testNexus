import pandas as pd
import numpy as np
from cleaning import auto_clean

def test_auto_clean_basic_behavior():
    df = pd.DataFrame({
        "A": ["  hello  ", "world", ""],
        "B": ["1", "2", "3"],
        "C": ["2024-01-01", "not-a-date", "2024-02-01"],
        "D": [np.nan, np.nan, np.nan]
    })

    cleaned = auto_clean(df)

    assert cleaned["A"].iloc[0] == "hello"
    assert pd.isna(cleaned["A"].iloc[2])
    assert cleaned["B"].dtype.kind in ("i", "f")
    assert pd.api.types.is_datetime64_any_dtype(cleaned["C"])
    assert "D" not in cleaned.columns

def test_auto_clean_drop_duplicates():
    df = pd.DataFrame({"A": ["x", "x"], "B": [1, 1]})
    cleaned = auto_clean(df)
    assert len(cleaned) == 1
