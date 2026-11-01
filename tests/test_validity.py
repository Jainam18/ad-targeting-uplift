import numpy as np
import pandas as pd
from scipy import stats

from uplift.validity import balance_auc, exposure_by_arm, smd_table, srm_test

FAST_HGB = {"max_iter": 30, "random_state": 42}


def test_srm_passes_correct_split_and_matches_scipy():
    res = srm_test(850_400, 149_600)
    ref = stats.chisquare([850_400, 149_600], f_exp=[850_000, 150_000])
    assert not res["flag"]
    assert np.isclose(res["p_value"], ref.pvalue)
    assert np.isclose(res["deviation_pp"], 0.04)


def test_srm_flags_broken_split():
    res = srm_test(80_000, 20_000)
    assert res["flag"]
    assert res["deviation_pp"] < -4.9


def test_srm_chi2_equals_squared_z():
    res = srm_test(850_400, 149_600)
    assert np.isclose(res["chi2"], res["deviation_in_se"] ** 2)


def _rct(n=200_000, shift=0.0, seed=42):
    rng = np.random.default_rng(seed)
    t = (rng.random(n) < 0.85).astype(np.int8)
    df = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n), "treatment": t})
    df.loc[df.treatment == 1, "f1"] += shift
    return df


def test_smd_near_zero_when_balanced():
    tab = smd_table(_rct(), ["f0", "f1"])
    assert (tab["abs_smd"] < 0.02).all()
    assert (tab["label"] == "balanced").all()


def test_smd_detects_known_shift():
    tab = smd_table(_rct(shift=0.2), ["f0", "f1"])
    assert abs(tab.loc["f1", "smd"] - 0.2) < 0.02
    assert tab.loc["f1", "label"] == "material"
    assert tab.loc["f0", "label"] == "balanced"


def test_balance_auc_near_half_under_randomization():
    df = _rct(n=50_000)
    res = balance_auc(df[["f0", "f1"]], df["treatment"], model_params=FAST_HGB)
    assert abs(res["mean_auc"] - 0.5) < 0.02
    assert not res["flag"]


def test_balance_auc_flags_feature_dependent_assignment():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50_000, 2))
    t = (rng.random(50_000) < 1 / (1 + np.exp(-(1.7 + 1.0 * x[:, 0])))).astype(int)
    res = balance_auc(x, t, model_params=FAST_HGB)
    assert res["mean_auc"] > 0.6
    assert res["flag"]


def test_exposure_flags_control_contamination():
    clean = pd.DataFrame({"treatment": [0, 1], "n": [100, 900], "exposures": [0, 30]})
    dirty = clean.assign(exposures=[2, 30])
    assert not exposure_by_arm(clean)["flag"]
    assert exposure_by_arm(clean)["treatment"]["exposure_rate"] == 30 / 900
    assert exposure_by_arm(dirty)["flag"]
