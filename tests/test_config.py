from uplift import config


def test_features_are_only_pretreatment_covariates():
    assert config.FEATURES == tuple(f"f{i}" for i in range(12))
    post_treatment = {
        config.TREATMENT,
        config.EXPOSURE,
        config.PRIMARY_OUTCOME,
        config.SECONDARY_OUTCOME,
    }
    assert post_treatment.isdisjoint(config.FEATURES)


def test_split_fractions_sum_to_one():
    assert abs(sum(config.SPLIT_FRACTIONS.values()) - 1.0) < 1e-12


def test_hyperparameters_use_project_seed():
    assert config.HGB_PARAMS["random_state"] == config.SEED == 42
