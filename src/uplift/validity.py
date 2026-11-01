"""Experiment validity checks (plan section 4). None of these look at outcomes."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from uplift.config import (
    BALANCE_AUC_FLAG,
    BALANCE_CV_FOLDS,
    HGB_PARAMS,
    INTENDED_TREATMENT_SHARE,
    SEED,
    SMD_MATERIAL,
    SMD_MINOR,
    SRM_P_THRESHOLD,
)


def srm_test(
    n_treatment: int,
    n_control: int,
    expected_share: float = INTENDED_TREATMENT_SHARE,
    p_threshold: float = SRM_P_THRESHOLD,
) -> dict[str, Any]:
    """Chi-square goodness of fit of arm counts against the intended split."""
    n = n_treatment + n_control
    expected = [expected_share * n, (1 - expected_share) * n]
    chi2, p = stats.chisquare([n_treatment, n_control], f_exp=expected)
    share = n_treatment / n
    se = np.sqrt(expected_share * (1 - expected_share) / n)
    return {
        "n_treatment": int(n_treatment),
        "n_control": int(n_control),
        "observed_share": share,
        "expected_share": expected_share,
        "deviation_pp": 100 * (share - expected_share),
        "deviation_in_se": (share - expected_share) / se,
        "chi2": float(chi2),
        "p_value": float(p),
        "flag": bool(p < p_threshold),
    }


def smd_table(
    df: pd.DataFrame,
    features: Sequence[str],
    treatment: str = "treatment",
    material: float = SMD_MATERIAL,
    minor: float = SMD_MINOR,
) -> pd.DataFrame:
    """Standardized mean difference per feature: (mean_t - mean_c) / sqrt((var_t + var_c) / 2)."""
    is_t = df[treatment].to_numpy() == 1
    rows = []
    for f in features:
        x = df[f].to_numpy(dtype=np.float64)
        xt, xc = x[is_t], x[~is_t]
        mt, mc = xt.mean(), xc.mean()
        vt, vc = xt.var(ddof=1), xc.var(ddof=1)
        pooled_sd = np.sqrt((vt + vc) / 2)
        smd = (mt - mc) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append(
            {
                "feature": f,
                "mean_t": mt,
                "mean_c": mc,
                "sd_t": np.sqrt(vt),
                "sd_c": np.sqrt(vc),
                "smd": smd,
            }
        )
    out = pd.DataFrame(rows).set_index("feature")
    out["abs_smd"] = out["smd"].abs()
    out["label"] = np.select(
        [out["abs_smd"] > material, out["abs_smd"] > minor], ["material", "minor"], "balanced"
    )
    return out


def balance_auc(
    X: pd.DataFrame | np.ndarray,
    t: pd.Series | np.ndarray,
    n_folds: int = BALANCE_CV_FOLDS,
    seed: int = SEED,
    flag_above: float = BALANCE_AUC_FLAG,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cross-validated AUC of a classifier predicting treatment from pre-treatment features."""
    X = np.asarray(X)
    t = np.asarray(t)
    params = dict(HGB_PARAMS if model_params is None else model_params)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in folds.split(X, t):
        model = HistGradientBoostingClassifier(**params)
        model.fit(X[train_idx], t[train_idx])
        aucs.append(roc_auc_score(t[test_idx], model.predict_proba(X[test_idx])[:, 1]))
    mean_auc = float(np.mean(aucs))
    return {
        "fold_aucs": [float(a) for a in aucs],
        "mean_auc": mean_auc,
        "sd_auc": float(np.std(aucs, ddof=1)),
        "flag_threshold": flag_above,
        "flag": bool(mean_auc > flag_above),
    }


def exposure_by_arm(arm_counts: pd.DataFrame) -> dict[str, Any]:
    """Exposure rate per arm from a table with columns treatment, n, exposures."""
    counts = arm_counts.set_index("treatment")
    arms = {
        name: {
            "n": int(counts.loc[code, "n"]),
            "exposed": int(counts.loc[code, "exposures"]),
            "exposure_rate": float(counts.loc[code, "exposures"] / counts.loc[code, "n"]),
        }
        for name, code in (("treatment", 1), ("control", 0))
    }
    return {**arms, "flag": arms["control"]["exposed"] > 0}
