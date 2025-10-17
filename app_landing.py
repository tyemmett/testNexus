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
                You are an assistant tax consultant estimating engagement hours based on nexus requirements for small businesses.
                The dataset has {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}.

                Here is a sample of the data:
                ```
                {sample_csv}
                ```

                Provide a concise summary of:
                1. Which nexus thresholds appear to be met (based on any columns indicating thresholds or boolean flags).
                2. Which states are within 10% of thresholds and might need follow-up. 
                3. What questions an analyst should consider before accepting or adjusting hours estimates. These should be state-specific recommendations.
                4. Any general data quality issues or missing fields worth noting.
                Keep it concise and analytical, focused on nexus interpretation.
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    messages=[{"role": "user", "content": summary_prompt}],
                )

                st.success("Analysis complete")
                st.markdown("### OpenAI Analysis Summary")
                st.markdown("FOR EDUCATIONAL PURPOSES ONLY")
                st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"Could not read the CSV: {e}")

st.divider()

