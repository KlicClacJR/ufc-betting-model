"""Compatibility wrapper for the UFC modeling workflow.

The project now exposes grouped modules under `src.modeling`, `src.odds`, and
`src.audits`, while preserving the older `src.train_baseline_models` import
surface so existing notebooks continue to run unchanged.
"""

from src.config import *  # noqa: F401,F403
from src.modeling.logistic import (  # noqa: F401
    run_validation_workflow,
    run_final_logistic_evaluation,
    train_and_evaluate_feature_set,
)
from src.modeling.random_forest import (  # noqa: F401
    run_final_random_forest_evaluation,
    run_random_forest_baseline,
    run_random_forest_random_search,
)
from src.modeling.xgboost_model import (  # noqa: F401
    run_final_xgboost_evaluation,
    run_xgboost_baseline,
    run_xgboost_random_search,
)
from src.modeling.walk_forward import run_walk_forward_market_research  # noqa: F401
from src.modeling.evaluation import build_final_model_test_comparison, build_final_model_comparison_summary  # noqa: F401
from src.odds.odds_ingestion import run_odds_ingestion_pipeline  # noqa: F401
from src.odds.betting_backtest import run_betting_backtest, run_random_forest_betting_backtest  # noqa: F401
from src.odds.betting_diagnostics import run_betting_failure_diagnostics  # noqa: F401
from src.audits.audit_modeling import run_modeling_audit  # noqa: F401
from src._pipeline_impl import run_final_model_family_comparison, main  # noqa: F401

__all__ = [
    'run_validation_workflow',
    'run_final_logistic_evaluation',
    'run_random_forest_baseline',
    'run_random_forest_random_search',
    'run_xgboost_baseline',
    'run_xgboost_random_search',
    'run_walk_forward_market_research',
    'run_odds_ingestion_pipeline',
    'run_betting_backtest',
    'run_random_forest_betting_backtest',
    'run_betting_failure_diagnostics',
    'run_modeling_audit',
    'run_final_random_forest_evaluation',
    'run_final_xgboost_evaluation',
    'run_final_model_family_comparison',
    'train_and_evaluate_feature_set',
    'main',
]

if __name__ == '__main__':
    main()
