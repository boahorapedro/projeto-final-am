"""Executa o pipeline completo: para cada dataset, treina e avalia todos os modelos.

Itera sobre 9 datasets do TabArena (3 small + 3 medium + 3 large), treina os
3 baselines (LightGBM/XGBoost/CatBoost via pytabkit) e, opcionalmente, o
modelo do grupo (`build_group_model`).

Uso:
    python -m src.pipeline.run_all --seed 42 --output results/raw.csv
    python -m src.pipeline.run_all --seed 42 --include-group-model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.load_tabarena import RECOMMENDED_TASK_IDS, load_task
from src.models.baselines import BASELINE_FACTORIES
from src.models.group_model import build_group_model
from src.pipeline.evaluate import fit_predict_evaluate
from src.pipeline.split import stratified_split


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

        factories: dict[str, callable] = dict(BASELINE_FACTORIES)
        if args.include_group_model:
            factories["group_model"] = build_group_model

        for model_name, factory in factories.items():
            estimator = factory(args.seed)
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
