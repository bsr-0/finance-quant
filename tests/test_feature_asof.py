import pandas as pd

from pipeline.features.feature_asof import align_features_labels, feature_asof


def test_feature_asof_shifts():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    shifted = feature_asof(df, horizon=1)
    assert shifted.iloc[0].isna().all()
    assert shifted.iloc[1]["a"] == 1
    assert shifted.iloc[2]["b"] == 20


def test_align_features_labels_drops_na():
    df = pd.DataFrame({"a": [1, 2, 3], "label": [0.1, 0.2, 0.3]})
    out = align_features_labels(df, label_col="label", horizon=1)
    assert len(out) == 2
    assert "label" in out.columns
