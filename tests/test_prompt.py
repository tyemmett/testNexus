from prompt import build_analysis_prompt

def test_build_analysis_prompt_contains_values():
    df_regression_input = {
        "states": 12,
        "channels": 3,
        "totalemployees": 150
    }

    class MockDF:
        columns = ["State", "Sales"]
        def __len__(self): return 100

    sample_csv = "State,Sales\nAR,1000\nMO,2000"

    prompt = build_analysis_prompt(df_regression_input, MockDF(), sample_csv)

    assert isinstance(prompt, str)
    assert "12" in prompt
    assert "3" in prompt
    assert "150" in prompt
    assert "AR,1000" in prompt
    assert "State" in prompt
    assert "Sales" in prompt
