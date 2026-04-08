"""
新患者死亡概率预测脚本

用法示例：
python3 new_patient_prediction.py
python3 new_patient_prediction.py --input new_patient_example.csv
python3 new_patient_prediction.py --input my_patient.csv --output results/my_prediction.csv
"""

import argparse
import os
from typing import Optional

import pandas as pd
from joblib import load

from classification_prediction import (
    BEST_MODEL_ARTIFACT_PATH,
    BINARY_FEATURE_COLUMNS,
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_and_save_best_model_artifact,
)

EXPERIMENT1_RESULT_PATH = "results/experiment1_model_comparison.csv"


DEFAULT_PATIENT = {
    "age": 68,
    "anaemia": 1,
    "creatinine_phosphokinase": 250,
    "diabetes": 0,
    "ejection_fraction": 25,
    "high_blood_pressure": 1,
    "platelets": 210000,
    "serum_creatinine": 1.8,
    "serum_sodium": 132,
    "sex": 1,
    "smoking": 0,
    "time": 90,
}


def parse_args():
    parser = argparse.ArgumentParser(description="预测新患者心衰死亡概率")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="包含 12 个特征列的 CSV 文件路径；若不提供则使用内置示例患者。"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/new_patient_prediction_result.csv",
        help="预测结果 CSV 保存路径。"
    )
    return parser.parse_args()


def ensure_artifact():
    if not os.path.exists(BEST_MODEL_ARTIFACT_PATH):
        if not os.path.exists(EXPERIMENT1_RESULT_PATH):
            raise FileNotFoundError(
                "缺少结果文件 results/experiment1_model_comparison.csv，"
                "请先运行 classification_prediction.py 生成实验结果。"
            )
        exp1_df = pd.read_csv(EXPERIMENT1_RESULT_PATH)
        best_result = exp1_df.iloc[0]
        build_and_save_best_model_artifact(
            best_model_name=best_result["Model"],
            reference_auc=best_result["AUC"],
            save_path=BEST_MODEL_ARTIFACT_PATH
        )
    return load(BEST_MODEL_ARTIFACT_PATH)


def load_patient_features(input_path: Optional[str]) -> pd.DataFrame:
    if input_path is None:
        patient_df = pd.DataFrame([DEFAULT_PATIENT])
    else:
        patient_df = pd.read_csv(input_path)

    missing_cols = [col for col in FEATURE_COLUMNS if col not in patient_df.columns]
    if missing_cols:
        raise ValueError(f"输入缺少必要字段：{missing_cols}")

    patient_df = patient_df[FEATURE_COLUMNS].copy()

    for col in BINARY_FEATURE_COLUMNS:
        invalid_mask = ~patient_df[col].isin([0, 1])
        if invalid_mask.any():
            raise ValueError(f"二分类字段 {col} 只能取 0 或 1。")

    return patient_df


def transform_features(patient_df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    transformed_df = patient_df.copy()
    scaler = artifact["scaler"]
    transformed_df[CONTINUOUS_FEATURE_COLUMNS] = scaler.transform(
        transformed_df[CONTINUOUS_FEATURE_COLUMNS]
    )
    return transformed_df


def risk_group(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.70:
        return "Medium"
    return "High"


def main():
    args = parse_args()
    artifact = ensure_artifact()
    patient_df = load_patient_features(args.input)
    transformed_df = transform_features(patient_df, artifact)

    model = artifact["model"]
    death_prob = model.predict_proba(transformed_df)[:, 1]
    death_pred = (death_prob >= 0.5).astype(int)

    result_df = patient_df.copy()
    result_df["predicted_death_probability"] = death_prob.round(6)
    result_df["predicted_outcome"] = death_pred
    result_df["risk_group"] = [risk_group(p) for p in death_prob]
    result_df["model_name"] = artifact["model_name"]
    result_df["reference_auc"] = artifact["reference_auc"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("新患者死亡概率预测结果：")
    print(result_df.to_string(index=False))
    print(f"\n结果已保存至：{args.output}")


if __name__ == "__main__":
    main()
