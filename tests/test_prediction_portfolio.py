import pandas as pd

from pipeline.eval.portfolio import ProbPortfolioConfig, generate_positions_from_probs


def test_prediction_positions_open_and_close():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "contract_id": ["c1"] * 10,
            "market_price": [0.4] * 10,
            "model_prob": [0.6] * 10,
        }
    )

    config = ProbPortfolioConfig(edge_threshold=0.05, notional_per_trade=100, holding_period_days=3)
    positions = generate_positions_from_probs(df, config)

    assert "c1" in positions.columns
    assert positions.iloc[0, 0] == 100
    assert positions.iloc[4, 0] == 100
    assert positions.iloc[5, 0] == 100
