"""Statistical analysis: parametric and nonparametric tests."""

from .nonparametric import (
    get_modal_category,
    kruskal_test,
    kruskal_test_multi,
    mannw_test,
    mcnemar_test,
    nonparametric_sig_test,
    p_value_sig,
    spearmanr,
)
from .parametric import paired_ttest, pearsonr_ci, z_test

__all__ = [
    # Nonparametric
    "kruskal_test",
    "kruskal_test_multi",
    "mannw_test",
    "spearmanr",
    "nonparametric_sig_test",
    "mcnemar_test",
    "get_modal_category",
    "p_value_sig",
    # Parametric
    "paired_ttest",
    "z_test",
    "pearsonr_ci",
]
