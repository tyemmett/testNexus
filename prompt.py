def build_analysis_prompt(df_regression_input, df_preview, sample_csv, predicted_hours=None):
    """Build a narrative-focused prompt for tax nexus analysis.

    The model should produce:
    - Executive summary of nexus position across all states
    - State-by-state breakdown with nexus status and key observations
    - Data quality assessment
    - Recommended action items prioritized by urgency
    - Estimated engagement hours with justification
    """

    predicted_block = f"Predicted engagement hours (regression model): {predicted_hours}" if predicted_hours is not None else "Predicted engagement hours: Not yet calculated"

    prompt = f"""
You are a tax nexus analyst reviewing state-level activity data for compliance assessment.

Regression inputs:
- Unique states: {df_regression_input.get('states')}
- Unique channels: {df_regression_input.get('channels')}
- Total employees: {df_regression_input.get('totalemployees')}

Dataset overview:
- Total rows: {len(df_preview)}
- Columns: {list(df_preview.columns)}

{predicted_block}

Sample data (first 10 rows):
```
{sample_csv}
```

Provide a comprehensive narrative analysis covering:

1. **Executive Summary**: Brief overview of overall nexus risk and urgency level

2. **State-by-State Analysis**: For each state in the data, provide:
   - Nexus position (met, approaching, or not close)
   - Key drivers (revenue, employees, transaction volume, etc.)
   - Risk level and urgency
   - Specific recommendations

3. **Data Quality Notes**: Any missing information, anomalies, or data gaps that affect analysis confidence

4. **Priority Action Items**: Concrete next steps ranked by urgency, such as:
   - Registration requirements
   - Additional data collection needed
   - Filing setup or system configuration
   - Compliance monitoring

5. **Engagement Hours Estimate**: Justify the predicted hours (or provide your own estimate if none given) by breaking down effort across tasks like initial review, research, setup, and ongoing compliance.

Guidelines:
- Professional, clear tone suitable for client delivery
- Be specific about thresholds and requirements where applicable
- Highlight urgent items clearly
- Treat all data as hypothetical for educational purposes
- Do not hallucinate states not present in the dataset
"""
    return prompt
