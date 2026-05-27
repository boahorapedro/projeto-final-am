"""Executa o pipeline completo: para cada dataset, treina e avalia todos os modelos.

Itera sobre 9 datasets do TabArena (3 small + 3 medium + 3 large), treina os
3 baselines (LightGBM/XGBoost/CatBoost via pytabkit) e, opcionalmente, o
modelo do grupo (`build_group_model`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from data.load_tabarena import RECOMMENDED_TASK_IDS, load_task
from src.models.baselines import BASELINE_FACTORIES
from src.models.group_model import build_group_model
from src.pipeline.evaluate import fit_predict_evaluate
from src.pipeline.split import stratified_split
from src.pipeline.tune import tune 


def xrfm_search_space(trial):
    """Espaço de busca do Optuna específico para o xRFM."""
    return {
        'iters': trial.suggest_int('iters', 2, 5),
        'bandwidth': trial.suggest_float('bandwidth', 1.0, 20.0, log=True),
        'exponent': trial.suggest_float('exponent', 0.5, 1.5),
        'reg': trial.suggest_float('reg', 1e-4, 1e-2, log=True)
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw.csv"),
        help="caminho do CSV de saída",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="opcional: lista de task IDs do OpenML; se omitido, usa RECOMMENDED_TASK_IDS",
    )
    parser.add_argument(
        "--include-group-model",
        action="store_true",
        help="se passado, inclui o modelo do grupo (build_group_model)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    task_ids = args.task_ids if args.task_ids else RECOMMENDED_TASK_IDS

    rows: list[dict] = []
    for task_id in task_ids:
        ds = load_task(task_id)
        X_train, X_test, y_train, y_test = stratified_split(ds.X, ds.y, seed=args.seed)

        # Garantir cópias para não alterar os dados originais guardados em cache
        X_train = X_train.copy()
        X_test = X_test.copy()

        # --- IMPUTAÇÃO INTELIGENTE VIA PANDAS ---
        # Filtra apenas colunas numéricas (contínuas) para calcular a mediana
        # Deixa colunas de texto (categóricas strings como 'female') totalmente intactas
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            median_val = X_train[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        factories: dict[str, callable] = dict(BASELINE_FACTORIES)
        if args.include_group_model:
            factories["group_model"] = build_group_model

        for model_name, factory in factories.items():
            
            # Executa a busca se for o modelo do grupo
            if model_name == "group_model":
                print(f"[{ds.name}] Rodando Optuna para o xRFM (pode demorar)...")
                best_params, _ = tune(
                    estimator_factory=lambda params: factory(args.seed, params),
                    search_space=xrfm_search_space,
                    X=X_train,
                    y=y_train,
                    seed=args.seed,
                    n_trials=15,   
                    cv_folds=3     
                )
                estimator = factory(args.seed, best_params)
            else:
                estimator = factory(args.seed)

            # Avaliação do modelo (sem o make_pipeline problemático)
            metrics = fit_predict_evaluate(estimator, X_train, y_train, X_test, y_test)
            
            row = {"task_id": task_id, "dataset": ds.name, "model": model_name}
            row.update(metrics.to_dict())
            rows.append(row)
            
            print(
                f"[{ds.name}] {model_name}: AUC={metrics.auc_ovo:.4f}, "
                f"ACC={metrics.accuracy:.4f}, time={metrics.fit_time_s + metrics.predict_time_s:.1f}s"
            )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nResultados gravados em {args.output}")


if __name__ == "__main__":
    main()