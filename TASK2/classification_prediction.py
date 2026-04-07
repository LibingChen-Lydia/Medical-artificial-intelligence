"""
心衰患者生存预测（二分类）完整建模代码
任务目标：预测 DEATH_EVENT（0=存活，1=死亡）

实验包含两部分：
1. Experiment 1：模型对比（统一使用 scaled 数据）
2. Experiment 2：数据敏感性分析（RF / XGB 在 raw、winsorized、scaled 上比较）
"""

import os
import random
import warnings
from typing import Dict, Tuple, List
from matplotlib import font_manager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# xgboost
from xgboost import XGBClassifier

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

RESULT_DIR = "results"
FIG_DIR = "figures"

def set_plot_style():
    from matplotlib import font_manager

    preferred_fonts = [
        "Times New Roman",
        "STIX Two Text",
        "Georgia",
        "Cambria",
        "Libertinus Serif",
        "Linux Libertine O",
        "DejaVu Serif",
    ]

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    chosen_font = None
    for f in preferred_fonts:
        if f in available_fonts:
            chosen_font = f
            break

    if chosen_font is None:
        chosen_font = "DejaVu Serif"

    print("Using font:", chosen_font)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [chosen_font],
        "mathtext.fontset": "stix",

        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,

        "axes.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

warnings.filterwarnings("ignore")


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "DEATH_EVENT"

RAW_PATH = "heart_failure_clinical_records_dataset.csv"
SCALED_PATH = "heart_failure_clinical_records_dataset_scaled.csv"
WINSORIZED_PATH = "heart_failure_clinical_records_dataset_winsorized.csv"


def set_seed(seed: int = 42):
    """固定随机种子，保证结果尽量可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了更强的可复现性（可能稍微影响速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(file_path: str, target_col: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    """
    加载数据，并拆分为特征 X 和标签 y
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件：{file_path}")

    df = pd.read_csv(file_path)

    if target_col not in df.columns:
        raise ValueError(f"目标列 {target_col} 不在数据中，请检查文件。")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    return X, y


def get_split_indices(y: pd.Series,
                      test_size: float = TEST_SIZE,
                      random_state: int = RANDOM_STATE):
    """
    只基于样本索引做一次划分，确保不同数据版本使用完全相同的 train/test 划分
    """
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return train_idx, test_idx


def apply_indices_split(X: pd.DataFrame,
                        y: pd.Series,
                        train_idx: np.ndarray,
                        test_idx: np.ndarray):
    """
    根据固定索引切分数据
    """
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    return X_train, X_test, y_train, y_test


class MLPClassifierTorch(nn.Module):
    """
    简单多层感知机：
    Input → Linear(32) → ReLU → Dropout(0.3)
          → Linear(16) → ReLU → Dropout(0.3)
          → Linear(1)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)


class TorchMLPWrapper:
    """
    为了和 sklearn 模型接口风格统一，封装一个 fit / predict_proba / predict
    """
    def __init__(self,
                 input_dim: int,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 80,
                 random_state: int = 42):
        self.input_dim = input_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = MLPClassifierTorch(input_dim=input_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X_train, y_train):
        set_seed(self.random_state)

        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"[MLP] Start training on device: {self.device}")
        print(f"[MLP] Train samples: {len(dataset)}, Batch size: {self.batch_size}, Epochs: {self.epochs}")

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)

            avg_loss = epoch_loss / len(dataset)

            # 每 5 个 epoch 打印一次训练损失
            if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                print(f"[MLP] Epoch {epoch:>3d}/{self.epochs}, Loss = {avg_loss:.4f}")

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        # 仿照 sklearn 的 predict_proba 输出两列：[P(class=0), P(class=1)]
        probs_2d = np.vstack([1 - probs, probs]).T
        return probs_2d

    def predict(self, X, threshold: float = 0.5):
        probs = self.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds


def get_experiment1_models(input_dim: int) -> Dict[str, object]:
    """
    Experiment 1 使用的所有模型
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
        "MLP": TorchMLPWrapper(
            input_dim=input_dim,
            lr=0.001,
            batch_size=16,
            epochs=30,
            random_state=RANDOM_STATE
        )
    }
    return models


def get_experiment2_models() -> Dict[str, object]:
    """
    Experiment 2 仅使用 RF 和 XGB
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
    }
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str) -> Dict:
    """
    训练模型并输出各项指标
    """
    print(f"\n开始训练模型：{model_name}")

    # 训练
    model.fit(X_train, y_train)

    # 预测类别
    y_pred = model.predict(X_test)

    # 预测概率（用于 AUC 和 ROC）
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        raise AttributeError(f"{model_name} 没有 predict_proba 方法，无法计算 AUC。")

    # 指标计算
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    # ROC 曲线数据
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    result = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "AUC": auc,
        "fpr": fpr,
        "tpr": tpr
    }

    print(
        f"{model_name} | "
        f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, "
        f"F1={f1:.4f}, AUC={auc:.4f}"
    )

    return result


def plot_experiment1_roc(results: List[Dict], save_path: str = "experiment1_roc.png"):
    """
    将 Experiment 1 所有模型的 ROC 曲线画在一张图上
    """
    plt.figure(figsize=(9, 7))

    for res in results:
        plt.plot(res["fpr"], res["tpr"], lw=2,
                 label=f'{res["Model"]} (AUC={res["AUC"]:.3f})')

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Experiment 1: ROC Curves of All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nROC 图已保存至：{save_path}")


def plot_experiment2_auc_bar(results_df: pd.DataFrame,
                             save_path: str = "experiment2_auc_bar.png"):


    # ===== 数据整理 =====
    pivot_df = results_df.pivot(index="Data Version", columns="Model", values="AUC")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # ===== 柱子参数 =====
    x = np.arange(len(pivot_df.index))
    width = 0.22

    colors = ["#bb1e38", "#0E6DB3"]  # 蓝 + 橙
    hatches = ["/", "x"]

    for i, model in enumerate(pivot_df.columns):
        bars = ax.bar(
            x + i * width - width/2,
            pivot_df[model],
            width,
            label=model,
            color=colors[i],
            edgecolor="white",
            linewidth=0.8,
            hatch=hatches[i],
            zorder=2
        )

        # ===== 数值标注 =====
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    # ===== 坐标轴 =====
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Data Version")

    # ===== 网格（弱虚线）=====
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4)

    # ===== 去掉上右边框 =====
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ===== 标题 =====
    ax.set_title("AUC Comparison Across Data Versions", pad=12)

    # ===== 图例优化 =====
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray")

    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.legend(loc="upper right")
    plt.show()

    print(f"Experiment 2 柱状图已保存至：{save_path}")


def run_experiment1(train_idx, test_idx) -> pd.DataFrame:
    """
    所有模型统一使用 scaled 数据，并使用完全相同的 train/test split
    """
    print("\n" + "=" * 70)
    print("Experiment 1：模型对比（统一使用 scaled 数据）")
    print("=" * 70)

    X_scaled, y_scaled = load_dataset(SCALED_PATH, TARGET_COL)
    X_train, X_test, y_train, y_test = apply_indices_split(
        X_scaled, y_scaled, train_idx, test_idx
    )

    models = get_experiment1_models(input_dim=X_train.shape[1])

    results = []
    for model_name, model in models.items():
        res = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
        results.append(res)

    # 结果表
    result_df = pd.DataFrame([
        {
            "Model": r["Model"],
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1-score": r["F1-score"],
            "AUC": r["AUC"]
        }
        for r in results
    ])

    result_df = result_df.sort_values(by="AUC", ascending=False).reset_index(drop=True)

    print("\nExperiment 1：模型对比结果（按 AUC 降序）")
    print(result_df.round(4).to_string(index=False))

    # ROC 图
    plot_experiment1_roc(
        results,
        save_path=os.path.join(FIG_DIR, "experiment1_roc.png")
    )

    return result_df


def run_experiment2(train_idx, test_idx) -> pd.DataFrame:
    """
    仅使用 RF / XGB，分别在 raw / winsorized / scaled 上建模
    """
    print("\n" + "=" * 70)
    print("Experiment 2：数据敏感性分析（RF / XGB）")
    print("=" * 70)

    data_files = {
        "raw": RAW_PATH,
        "winsorized": WINSORIZED_PATH,
        "scaled": SCALED_PATH
    }

    models = get_experiment2_models()

    all_results = []

    for data_name, file_path in data_files.items():
        print(f"\n--- 当前数据版本：{data_name} ---")
        X, y = load_dataset(file_path, TARGET_COL)
        X_train, X_test, y_train, y_test = apply_indices_split(X, y, train_idx, test_idx)

        for model_name, model in models.items():
            res = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)

            all_results.append({
                "Data Version": data_name,
                "Model": model_name,
                "Accuracy": res["Accuracy"],
                "Precision": res["Precision"],
                "Recall": res["Recall"],
                "F1-score": res["F1-score"],
                "AUC": res["AUC"]
            })

    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values(by=["Model", "AUC"], ascending=[True, False]).reset_index(drop=True)

    print("\nExperiment 2：不同数据版本下 RF / XGB 对比结果")
    print(result_df.round(4).to_string(index=False))

    plot_experiment2_auc_bar(
        result_df,
        save_path=os.path.join(FIG_DIR, "experiment2_auc_bar.png")
    )

    return result_df



def main():
    set_seed(RANDOM_STATE)

    print("开始执行心衰患者生存预测建模任务...\n")

    set_plot_style()

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    X_scaled, y_scaled = load_dataset(SCALED_PATH, TARGET_COL)
    train_idx, test_idx = get_split_indices(
        y_scaled,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Experiment 1
    exp1_df = run_experiment1(train_idx, test_idx)

    # Experiment 2
    exp2_df = run_experiment2(train_idx, test_idx)

    # 保存结果表
    exp1_df.to_csv(
        os.path.join(RESULT_DIR, "experiment1_model_comparison.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    exp2_df.to_csv(
        os.path.join(RESULT_DIR, "experiment2_data_sensitivity.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("全部实验完成。")
    print("结果文件已保存：")
    print("1. experiment1_model_comparison.csv")
    print("2. experiment2_data_sensitivity.csv")
    print("3. figures/experiment1_roc.png")
    print("4. figures/experiment2_auc_bar.png")
    print("=" * 70)


if __name__ == "__main__":
    main()