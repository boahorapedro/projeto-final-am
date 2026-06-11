"""Busca de hiperparâmetros com Optuna em validação cruzada (CV) no treino."""
from __future__ import annotations

import traceback
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_N_TRIALS = 50
DEFAULT_CV_FOLDS = 5
DEFAULT_SCORING = "roc_auc_ovo"


def tune(
    estimator_factory: Callable[[dict[str, Any]], Any],
    search_space: Callable[[optuna.Trial], dict[str, Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42,
    n_trials: int = DEFAULT_N_TRIALS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
) -> tuple[dict[str, Any], float]:
    """Roda Optuna no espaço de busca passado e retorna (melhor_params, melhor_score)."""

    y = np.array(y)
    if np.issubdtype(y.dtype, np.floating):
        y = y.astype(int)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Guarda o último erro real para diagnóstico
    _last_error: list[str] = []

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        estimator = estimator_factory(params)

        try:
            scores = cross_val_score(
                estimator, X, y,
                scoring=scoring,
                cv=cv,
                n_jobs=1,
                error_score="raise",
            )
        except Exception as e:
            # ── NOVO: imprime o traceback COMPLETO do erro real ──────────────
            tb = traceback.format_exc()
            msg = f"[Optuna Trial {trial.number}] ERRO com params={params}:\n{tb}"
            print(msg)
            _last_error.clear()
            _last_error.append(msg)
            # ─────────────────────────────────────────────────────────────────
            raise optuna.exceptions.TrialPruned(
                f"Trial descartado por erro no CV: {e}"
            )

        mean_score = float(np.mean(scores))
        if np.isnan(mean_score):
            raise optuna.exceptions.TrialPruned("Score retornou NaN.")

        return mean_score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed:
        last_err = _last_error[0] if _last_error else "(sem informação de erro)"
        raise RuntimeError(
            "Nenhum trial completou com sucesso.\n"
            f"Último erro registrado:\n{last_err}\n\n"
            "Dica: verifique o traceback acima para a causa raiz."
        )

    best = max(completed, key=lambda t: t.value)
    return best.params, float(best.value)