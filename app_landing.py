# Purpose: A minimal Streamlit landing page with a CSV upload to get started. testing functions of streamlit.

from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Nexus Scout", page_icon="🧭", layout="centered")

#################################
# Header
#################################

st.title("🧭 Nexus Scout")
st.subheader("Upload your activity CSV to get started")
st.caption(
    "Expected columns: **date**, **jurisdiction** (or state), **hours**. "
    "You can change names later — this step is just to load a file and preview it."
)

################################
# Sample & Helper 
################################

# sample CSV for download
SAMPLE_CSV = """date,jurisdiction,hours
2025-06-01,AR,12
2025-06-05,AR,6
2025-07-15,MO,14
2025-08-20,OK,22
"""

col_a, col_b = st.columns([1,1])
with col_a:
    st.download_button(
        "Download sample CSV",
        data=SAMPLE_CSV.encode("utf-8"),
        file_name="nexus_sample.csv",
        mime="text/csv",
        help="Grab an example file to test the upload."
    )
with col_b:
    st.link_button(
        "View expected schema",
        url="#expected-schema",
        help="Jump to the short description below"
    )

st.divider()

#################################
# Uploader 
#################################
uploaded = st.file_uploader(
    "Drop a CSV here or click to browse",
    type=["csv"],
    accept_multiple_files=False,
)

if not uploaded:
    st.info("No file yet — upload a CSV to see a preview.")
else:
    # Basic ingestion 
    try:
        df = pd.read_csv(uploaded)
        # Make soft conversions for a friendlier preview; this is not strict validation
        lowered = {c.lower(): c for c in df.columns}
        # Show a soft mapping for common names
        schema_map = {}
        if "date" in lowered: schema_map[lowered["date"]] = "date"
        if "jurisdiction" in lowered: schema_map[lowered["jurisdiction"]] = "jurisdiction"
        if "state" in lowered and "jurisdiction" not in lowered:
            schema_map[lowered["state"]] = "jurisdiction"
        if "hours" in lowered: schema_map[lowered["hours"]] = "hours"

        df_preview = df.rename(columns=schema_map).copy()
        if "date" in df_preview.columns:
            df_preview["date"] = pd.to_datetime(df_preview["date"], errors="coerce")
        if "jurisdiction" in df_preview.columns:
            df_preview["jurisdiction"] = df_preview["jurisdiction"].astype(str)
        if "hours" in df_preview.columns:
            df_preview["hours"] = pd.to_numeric(df_preview["hours"], errors="coerce")

        st.success(f"Loaded {len(df):,} rows.")
        st.markdown("**Preview (first 100 rows)**")
        st.dataframe(df_preview.head(100), use_container_width=True)

        # Light status chips 
        cols = st.columns(3)
        cols[0].metric("Columns detected", len(df.columns))
        cols[1].metric("Rows detected", len(df))
        cols[2].metric("Nulls (any col)", int(df.isna().any(axis=1).sum()))

        st.divider()
        st.markdown("### Next step")
        st.caption(
            "Looks good? In the next iteration we’ll map your columns and move on to rules & visuals."
        )
        st.button("Continue (disabled for this landing‑page build)", disabled=True)

    except Exception as e:
        st.error(f"Could not read the CSV: {e}")

st.divider()
