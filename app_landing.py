from datetime import datetime
import pandas as pd
import streamlit as st
from openai import OpenAI

# Initialize the OpenAI client (make sure your OPENAI_API_KEY is set in environment)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])



st.set_page_config(page_title="Nexus Scout", page_icon="🧭", layout="centered")

#################################
# Header
#################################
st.title("🧭 Nexus Scout")
st.subheader("Upload your activity CSV to get started")
st.caption("Expected columns: PLACEHOLDER")

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
    try:
        df = pd.read_csv(uploaded)
        lowered = {c.lower(): c for c in df.columns}
        schema_map = {}
        if "date" in lowered: schema_map[lowered["date"]] = "date"
        if "jurisdiction" in lowered: schema_map[lowered["jurisdiction"]] = "jurisdiction"
        if "state" in lowered and "jurisdiction" not in lowered:
            schema_map[lowered["state"]] = "jurisdiction"

        df_preview = df.rename(columns=schema_map).copy()
        if "date" in df_preview.columns:
            df_preview["date"] = pd.to_datetime(df_preview["date"], errors="coerce")
        if "jurisdiction" in df_preview.columns:
            df_preview["jurisdiction"] = df_preview["jurisdiction"].astype(str)

        st.success(f"Loaded {len(df):,} rows.")
        st.markdown("**Preview (first 100 rows)**")
        st.dataframe(df_preview.head(100), use_container_width=True)

        cols = st.columns(3)
        cols[0].metric("Columns detected", len(df.columns))
        cols[1].metric("Rows detected", len(df))
        cols[2].metric("Nulls (any col)", int(df.isna().any(axis=1).sum()))

        st.divider()
        st.markdown("### Next step")
        st.caption("Click below to run initial analysis")

        if st.button("Continue", disabled=False):
            with st.spinner("Calling OpenAI API..."):
            # Limit rows for prompt safety
                sample_csv = df.head(10).to_csv(index=False)

                summary_prompt = f"""
                You are a tax nexus analyst reviewing state-level data to estimate engagement hours.
                You will receive a small dataset with one row per state.

                Dataset overview:
                - Total rows: {len(df)}
                - Columns: {list(df.columns)}

                Here is the data :
                ```
                {sample_csv}
                ```
                Your task:
                1. Review the data and interpret each state's nexus position using the columns provided 
               (e.g., TotalSales, Transactions, PhysicalFootprint, Employees, and thresholds).
                2. For each state, summarize:
                   - Whether the nexus threshold appears **met**, **within 10% of being met**, or **not close**.
                   - A short, state-specific recommendation (e.g., questions to ask, trends to check, 
                     or actions to confirm/monitor).
                3. Provide an overall summary:
                   - How many states are met / close / far.
                   - Where to prioritize analysis time.
                4. Mention any obvious data quality issues or missing columns.

                Guidelines:
                - Use clear, bullet-point summaries by state.
                - Keep your tone professional and analytical, like a consultant’s workpaper summary.
                - Assume this is mock data for planning, not real tax advice.
                - Questions and analysis should focus on the task of estimating hours for the engagement, not increasing sales, etc.
                - Give an estimate of hours for the company and explain why that estimate.
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    messages=[{"role": "user", "content": summary_prompt}],
                )

                st.success("Analysis complete")
                st.markdown("### OpenAI Analysis Summary")
                st.markdown("FOR EDUCATIONAL PURPOSES ONLY")
                st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"Could not read the CSV: {e}")

st.divider()

