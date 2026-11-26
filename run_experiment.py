import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from grader_contracts.ensembles import EnsembleInput
from ensemble_models import run_ensemble_experiment


def main():
    data_path = Path(__file__).with_name("german.csv")
    data = pd.read_csv(data_path, sep=";")

    target = data["Creditability"]
    features = data.drop(columns=["Creditability"])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    experiment_input = EnsembleInput(
        train_features=x_train,
        train_target=y_train,
        test_features=x_test,
        test_target=y_test,
        random_seed=42,
    )

    result = run_ensemble_experiment(experiment_input)

    metrics = {
        "model_name": result.model_name,
        "roc_auc": result.roc_auc,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
    }

    print("Результаты эксперимента")
    print("-----------------------")
    print(f"Модель:    {metrics['model_name']}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

    output_path = Path(__file__).with_name("experiment_results.json")
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    print(f"\nРезультаты сохранены в: {output_path.name}")


if __name__ == "__main__":
    main()
