import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from facedyn.normalisation import ZScoreShiftNormalizer


def test_fit_learns_pooled_mean_and_sd():
    values = [1.0, 3.0, 2.0, 8.0, 4.0, 4.0, 9.0]
    df = pd.DataFrame({"smth_x": values})
    normalizer = ZScoreShiftNormalizer().fit(df)

    assert normalizer.means_["smth_x"] == pytest.approx(np.mean(values))
    assert normalizer.sds_["smth_x"] == pytest.approx(np.std(values, ddof=1))


def test_constant_column_sd_guarded_to_one():
    df = pd.DataFrame({"smth_x": [5.0, 5.0, 5.0]})
    normalizer = ZScoreShiftNormalizer().fit(df)

    assert normalizer.sds_["smth_x"] == 1.0
    out = normalizer.transform(df)
    assert np.isfinite(out["smth_x"]).all()


def test_transform_output_min_is_zero():
    df = pd.DataFrame({"smth_x": [1.0, 3.0, 2.0, 8.0, 4.0, 4.0, 9.0]})
    normalizer = ZScoreShiftNormalizer().fit(df)
    out = normalizer.transform(df)

    assert out["smth_x"].min() == pytest.approx(0.0, abs=1e-9)


def test_shift_is_recomputed_independently_per_call():
    """The non-negativity shift is NOT a frozen fit-time parameter — it's
    recomputed fresh on whatever data transform() is called with. Proof:
    two datasets differing by a constant offset produce IDENTICAL output,
    because each gets independently anchored to its own minimum."""
    fit_data = pd.DataFrame({"smth_x": [0.0, 2.0, 4.0, 6.0, 8.0]})
    normalizer = ZScoreShiftNormalizer().fit(fit_data)

    b = pd.DataFrame({"smth_x": [10.0, 12.0, 14.0]})
    c = pd.DataFrame({"smth_x": [110.0, 112.0, 114.0]})  # b + 100

    out_b = normalizer.transform(b)
    out_c = normalizer.transform(c)

    pd.testing.assert_series_equal(
        out_b["smth_x"], out_c["smth_x"], check_exact=False, atol=1e-9
    )
    assert out_b["smth_x"].min() == pytest.approx(0.0, abs=1e-9)
    assert out_c["smth_x"].min() == pytest.approx(0.0, abs=1e-9)


def test_mean_and_sd_are_frozen_from_fit_not_recomputed():
    """Unlike the shift, mean/SD used for the z-score itself come from
    fit() and are NOT recomputed from the data passed to transform()."""
    fit_data = pd.DataFrame({"smth_x": [0.0, 2.0, 4.0, 6.0, 8.0]})
    new_data = pd.DataFrame({"smth_x": [10.0, 12.0, 14.0]})

    normalizer_fit_on_train = ZScoreShiftNormalizer().fit(fit_data)
    normalizer_fit_on_new = ZScoreShiftNormalizer().fit(new_data)

    out_using_train_params = normalizer_fit_on_train.transform(new_data)
    out_using_new_params = normalizer_fit_on_new.transform(new_data)

    assert not np.allclose(
        out_using_train_params["smth_x"], out_using_new_params["smth_x"]
    )


def test_nan_input_becomes_zero():
    df = pd.DataFrame({"smth_x": [1.0, 2.0, np.nan, 4.0]})
    normalizer = ZScoreShiftNormalizer().fit(df)
    out = normalizer.transform(df)

    assert out["smth_x"].iloc[2] == 0.0


def test_transform_before_fit_raises():
    df = pd.DataFrame({"smth_x": [1.0, 2.0, 3.0]})
    with pytest.raises(NotFittedError):
        ZScoreShiftNormalizer().transform(df)


def test_column_selection_by_pattern():
    df = pd.DataFrame({
        "smth_AU01_r": [1.0, 2.0, 3.0],
        "smth_AU12_r": [0.0, 1.0, 1.0],
        "frame": [1, 2, 3],  # not "smth_"-prefixed, should be excluded
    })
    normalizer = ZScoreShiftNormalizer().fit(df)
    assert set(normalizer.columns_) == {"smth_AU01_r", "smth_AU12_r"}
    assert "frame" not in normalizer.columns_


def test_explicit_columns_override_pattern():
    df = pd.DataFrame({
        "custom_signal": [1.0, 2.0, 3.0],
        "other": [4.0, 5.0, 6.0],
    })
    normalizer = ZScoreShiftNormalizer(columns=["custom_signal"]).fit(df)
    assert normalizer.columns_ == ["custom_signal"]
