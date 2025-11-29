from datetime import datetime
import os
import json
import pandas as pd
import streamlit as st
import numpy as np
from openai import OpenAI

from cleaning import auto_clean
from prompt import build_analysis_prompt   
from run_regression import train_and_save, load_model_bundle, predict_with_model
from pathlib import Path

#################################
# OpenAI Client Initialization
#################################
def get_openai_client():
    key = None
    try:
        # Prefer Streamlit secrets if available
        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        # st.secrets may not be configured; ignore
        pass
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)

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
    #################################
    # CSV Loading & Cleaning
    #################################
    # Load + clean only; handle read errors specifically
    try:
        df = pd.read_csv(uploaded)
    except Exception as csv_err:
        st.error(f"Failed to read CSV: {csv_err}")
        st.stop()

    df_clean = auto_clean(df)

    # (Moved) Regression inputs will be computed after normalization with flexible column mapping.

    #################################
    # NORMALIZATION
    #################################
    lowered = {c.lower(): c for c in df_clean.columns}
    schema_map = {}

    if "date" in lowered:
        schema_map[lowered["date"]] = "date"
    if "jurisdiction" in lowered:
        schema_map[lowered["jurisdiction"]] = "jurisdiction"
    if "state" in lowered and "jurisdiction" not in lowered:
        schema_map[lowered["state"]] = "jurisdiction"

    df_preview = df_clean.rename(columns=schema_map).copy()

    if "date" in df_preview.columns:
        df_preview["date"] = pd.to_datetime(df_preview["date"], errors="coerce")
    if "jurisdiction" in df_preview.columns:
        df_preview["jurisdiction"] = df_preview["jurisdiction"].astype(str)

    #################################
    # REGRESSION INPUTS (post-normalization)
    #################################
    preview_lowered = {c.lower(): c for c in df_preview.columns}
    state_like = preview_lowered.get("jurisdiction") or preview_lowered.get("state")
    channel_like = preview_lowered.get("business channel") or preview_lowered.get("channel")
    employees_like = (
        preview_lowered.get("number of employees")
        or preview_lowered.get("total employees")
        or preview_lowered.get("employees")
    )

    states_count = int(df_preview[state_like].nunique()) if state_like in df_preview else None
    channels_count = int(df_preview[channel_like].nunique()) if channel_like in df_preview else None
    total_employees = None
    if employees_like in df_preview:
        total_employees = int(pd.to_numeric(df_preview[employees_like], errors="coerce").fillna(0).sum())

    df_regression_input = {
        "states": states_count,
        "channels": channels_count,
        "totalemployees": total_employees,
    }

    #################################
    # Data Preview & Regression Inputs
    #################################
    st.success(f"Loaded {len(df_clean):,} cleaned rows.")
    st.markdown("### Cleaned Data Preview (first 100 rows)")
    st.dataframe(df_preview.head(100), use_container_width=True)

    st.json(df_regression_input)

    #################################
    # User Input for Regression Features
    #################################
    st.divider()
    st.markdown("### Additional Information for Prediction")
    st.caption("Please provide the following details to improve regression accuracy:")
    
    col1, col2 = st.columns(2)
    with col1:
        data_systems_input = st.number_input(
            "How many data systems?",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Number of different data systems or platforms used."
        )
    with col2:
        product_categories_input = st.number_input(
            "How many product categories?",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Number of distinct product categories sold."
        )

    #################################
    # Regression Prediction
    #################################
    st.divider()
    st.markdown("### Predict Engagement Hours (Regression)")
    st.caption("Uses the trained model from `nexus_hours_regression_data.csv`.")
    if st.button("Predict hours for uploaded CSV"):
        with st.spinner("Training/loading model and generating predictions..."):
            model_path = Path("model.joblib")
            if not model_path.exists():
                train_and_save(
                    csv_path=Path("nexus_hours_regression_data.csv"),
                    target="hours",
                    model_name="linear",
                    test_size=0.2,
                    random_state=42,
                    save_model=model_path,
                )

            bundle = load_model_bundle(model_path)
            feature_columns = bundle.get("feature_columns", [])

            def build_feature_row(smb_df: pd.DataFrame, data_sys: int, prod_cats: int) -> pd.DataFrame:
                lowered = {c.lower(): c for c in smb_df.columns}
                state_col = lowered.get("jurisdiction") or lowered.get("state")
                channel_col = lowered.get("business channel") or lowered.get("channel")
                employees_col = (
                    lowered.get("number of employees")
                    or lowered.get("total employees")
                    or lowered.get("employees")
                )
                states = int(smb_df[state_col].nunique()) if state_col else 0
                channels = int(smb_df[channel_col].nunique()) if channel_col else 0
                employees = int(pd.to_numeric(smb_df[employees_col], errors="coerce").fillna(0).sum()) if employees_col else 0
                derived = {
                    "states": states,
                    "channels": channels,
                    "employees": employees,
                    "data_systems": data_sys,
                    "nexus_events": 1,
                    "product_categories": prod_cats,
                    "automation_level": 1,
                }
                return pd.DataFrame([derived])

            try:
                feature_df = build_feature_row(df_preview, data_systems_input, product_categories_input)
                missing = [c for c in feature_columns if c not in feature_df.columns]
                if missing:
                    st.error(
                        "Cannot derive all required features from the uploaded CSV: "
                        + ", ".join(missing)
                    )
                    st.caption("Upload should include State, Business Channel, Number of Employees; other features use defaults.")
                    st.stop()

                feature_df = feature_df.reindex(columns=feature_columns)
                csv_bytes = feature_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download regression input CSV",
                    data=csv_bytes,
                    file_name="regression_input_export.csv",
                    mime="text/csv",
                )
                preds = predict_with_model(bundle, feature_df)
                st.success("Predictions generated.")
                st.markdown("#### Regression input derived from upload")
                st.dataframe(feature_df, use_container_width=True)
                predicted = float(preds[0])
                st.session_state["predicted_hours"] = predicted
                st.markdown("#### Predicted engagement hours")
                st.write({"predicted_hours": predicted})
            except Exception as e:
                st.error(f"Could not build regression features from the uploaded CSV: {e}")

    #################################
    # OpenAI Nexus Analysis
    #################################
    # Now that prediction can be computed, run OpenAI summary next
    st.divider()
    st.markdown("### Next step")
    st.caption("Click below to run initial analysis (includes predicted hours when available)")

    if st.button("Continue"):
        with st.spinner("Calling OpenAI API..."):
            client = get_openai_client()
            if client is None:
                st.error("Missing OpenAI API key. Set it in secrets or env.")
                st.stop()
            sample_csv = df_preview.head(10).to_csv(index=False)
            predicted_hours = st.session_state.get("predicted_hours")
            summary_prompt = build_analysis_prompt(
                df_regression_input=df_regression_input,
                df_preview=df_preview,
                sample_csv=sample_csv,
                predicted_hours=predicted_hours,
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    messages=[{"role": "user", "content": summary_prompt}],
                )
            except Exception as api_err:
                st.error(f"OpenAI request failed: {api_err}")
                st.stop()

        st.success("Analysis complete")
        st.markdown("### OpenAI Nexus Analysis")
        st.caption("FOR EDUCATIONAL PURPOSES ONLY")
        
        # Display the narrative response directly
        raw_content = response.choices[0].message.content.strip()
        st.markdown(raw_content)

    #################################
    # Nexus Rules Evaluation
    #################################
    # Nexus rules section (outside of prediction button)
    st.divider()
    st.markdown("### Check Nexus Rules by State")
    st.caption("Rules are loaded from `nexus_rules.json`. Registration is inferred from data (doing business or employees > 0).")

    RULES_PATH = Path("nexus_rules.json")
    try:
        rules_text = RULES_PATH.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"Could not read {RULES_PATH}: {e}")
        rules_text = "{}"

    # Column mapping helpers
    lowered_cols = {c.lower(): c for c in df_preview.columns}
    state_col = lowered_cols.get("jurisdiction") or lowered_cols.get("state")

    # Evaluate rules
    if st.button("Evaluate Nexus Rules"):
        try:
            rules = json.loads(rules_text)
        except Exception as e:
            st.error(f"Invalid JSON for rules: {e}")
            st.stop()

        name_to_code = {"Arkansas": "AR", "Texas": "TX", "Oklahoma": "OK", "Missouri": "MO"}
        revenue_col = lowered_cols.get("revenue")
        employees_col = lowered_cols.get("number of employees")
        channel_col = lowered_cols.get("business channel")

        df_work = df_preview.copy()
        if revenue_col:
            df_work[revenue_col] = pd.to_numeric(df_work[revenue_col], errors="coerce").fillna(0)
        if employees_col:
            df_work[employees_col] = pd.to_numeric(df_work[employees_col], errors="coerce").fillna(0)

        def derive_state_facts(abbrev: str) -> dict:
            if not state_col:
                return {}
            mask = df_work[state_col].astype(str).str.upper() == abbrev
            rev = float(df_work.loc[mask, revenue_col].sum()) if revenue_col else 0.0
            emp = int(df_work.loc[mask, employees_col].sum()) if employees_col else 0
            doing_business = bool(mask.any())
            has_prop_emp = emp > 0
            registered = doing_business or has_prop_emp
            return {
                f"revenue_{abbrev}": rev,
                f"has_property_or_employees_{abbrev}": has_prop_emp,
                f"doing_business_{abbrev}": doing_business,
                "registered_in_state": registered,
            }

        def apply_op(lhs, op, rhs):
            if op == "==":
                return lhs == rhs
            if op == "!=":
                return lhs != rhs
            if op == ">":
                return float(lhs) > float(rhs)
            if op == ">=":
                return float(lhs) >= float(rhs)
            if op == "<":
                return float(lhs) < float(rhs)
            if op == "<=":
                return float(lhs) <= float(rhs)
            return False

        results = []
        for state_name, cfg in rules.items():
            abbrev = name_to_code.get(state_name)
            if not abbrev:
                continue
            facts = derive_state_facts(abbrev)
            rule_outcomes = []
            for r in cfg.get("rules", []):
                field = r.get("field")
                op = r.get("operator")
                expected = r.get("value")
                lhs = facts.get(field)
                passed = apply_op(lhs, op, expected) if lhs is not None else False
                rule_outcomes.append({
                    "id": r.get("id"),
                    "field": field,
                    "lhs": lhs,
                    "operator": op,
                    "rhs": expected,
                    "passed": passed,
                })
            overall = all(ro["passed"] for ro in rule_outcomes) if rule_outcomes else False
            failed = [ro["id"] for ro in rule_outcomes if not ro["passed"]]
            results.append({
                "state": state_name,
                "abbrev": abbrev,
                "nexus_met": overall,
                "failed_rules": ", ".join(failed) if failed else "",
                **{f"value_{ro['field']}": ro["lhs"] for ro in rule_outcomes},
            })

        st.markdown("#### Nexus evaluation results")
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("No results to display. Ensure your upload has a state column and rules reference known states.")

st.divider()
