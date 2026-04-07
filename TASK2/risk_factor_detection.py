"""
心衰患者生存预测项目：因子检测（Risk Factor Detection / Feature Engineering）

任务内容：
1. 读取数据
2. 区分连续变量与二分类变量
3. 绘制相关性热力图
4. 对连续变量进行独立样本 T 检验
5. 对二分类变量进行卡方检验
6. 使用随机森林计算特征重要性
7. 综合统计检验与模型重要性，输出排名前三的危险因子
8. 保存图像与结果文件

说明：
- 医学数据分析中，统计显著性与模型重要性是两个不同角度，应结合解释
- 连续变量若不满足正态性，后续也可补充 Mann–Whitney U 检验做稳健性分析
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import font_manager
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# 图像风格设置：尽量干净、克制、适合论文插图
sns.set_style("white")
preferred_fonts = [
    "Times New Roman",
    "STIX Two Text",
    "Georgia",
    "Cambria",
    "Libertinus Serif",
    "Linux Libertine O",
    "DejaVu Serif",
]

chosen_font = None
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
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
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9.5,
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.unicode_minus": False,
})


# 随机种子，保证结果可复现
RANDOM_STATE = 42

# 文件路径
FILE_PATH = "heart_failure_clinical_records_dataset.csv"
FIG_DIR = "figures"
RESULT_DIR = "results"
TABLES_DIR = "tables"


def print_section(title: str):
    """打印分节标题，方便阅读输出。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_directories():
    """如果保存结果的文件夹不存在，则自动创建。"""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def load_data(file_path: str) -> pd.DataFrame:
    """读取数据文件。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件：{file_path}，请确认数据文件在当前目录下。")
    df = pd.read_csv(file_path)
    return df


def classify_variables(df: pd.DataFrame, target_col: str = "DEATH_EVENT"):
    """
    自动区分连续变量与二分类变量。

    规则：
    - 数值型变量中，若去重后仅有2个取值，则视为二分类变量
    - 其余数值型变量视为连续变量

    说明：
    - 在该数据集中，anaemia、diabetes、high_blood_pressure、sex、smoking 等
      通常是二分类变量，虽然编码为0/1，但本质上是类别而非连续量
    """
    if target_col not in df.columns:
        raise ValueError(f"数据中未找到目标变量：{target_col}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    binary_cols = []
    continuous_cols = []

    for col in numeric_cols:
        unique_values = sorted(df[col].dropna().unique().tolist())
        if len(unique_values) == 2:
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    # 目标变量不作为自变量参与后续检验
    binary_feature_cols = [col for col in binary_cols if col != target_col]

    return continuous_cols, binary_cols, binary_feature_cols


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str):
    """
    对所有数值变量绘制相关性热力图，并保存图片。
    """
    print_section("1. 相关性热力图")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    ax.set_title("Correlation Heatmap of Numeric Variables", fontsize=13, pad=12)

    # 去掉上边框和右边框，让图像更干净
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"相关性热力图已保存到：{save_path}")


def t_test_continuous_variables(df: pd.DataFrame,
                                continuous_cols: list,
                                target_col: str = "DEATH_EVENT") -> pd.DataFrame:
    """
    将样本按 DEATH_EVENT 分为两组，对每个连续变量进行独立样本 T 检验。

    输出：
    - 变量名
    - 两组均值
    - t值
    - p值

    说明：
    - 这里使用 Welch's t-test（equal_var=False），对两组方差不齐更稳健
    - 若连续变量分布偏态较强，后续也可考虑 Mann–Whitney U 检验作为补充
    """
    print_section("2. 连续变量独立样本 T 检验")

    group_0 = df[df[target_col] == 0]
    group_1 = df[df[target_col] == 1]

    results = []

    for col in continuous_cols:
        # 删除缺失值后再做检验（该数据集通常无缺失，但这里保留稳健写法）
        x0 = group_0[col].dropna()
        x1 = group_1[col].dropna()

        # 如果某列无法形成两组有效数据，则跳过
        if len(x0) < 2 or len(x1) < 2:
            # 这种情况下无法进行稳定的T检验
            continue

        # 使用 Welch's t-test
        t_stat, p_value = ttest_ind(x0, x1, equal_var=False)

        results.append({
            "variable": col,
            "mean_DEATH_EVENT_0": round(x0.mean(), 4),
            "mean_DEATH_EVENT_1": round(x1.mean(), 4),
            "t_statistic": round(t_stat, 4),
            "p_value": p_value
        })

    results_df = pd.DataFrame(results).sort_values(by="p_value", ascending=True).reset_index(drop=True)

    print("连续变量 T 检验结果（按 p 值升序）：")
    print(results_df)

    return results_df


def chi_square_binary_variables(df: pd.DataFrame,
                                binary_feature_cols: list,
                                target_col: str = "DEATH_EVENT") -> pd.DataFrame:
    """
    对每个二分类变量与 DEATH_EVENT 之间进行卡方检验。

    输出：
    - 变量名
    - 卡方统计量
    - p值

    说明：
    - DEATH_EVENT 本身是目标变量，不与自己做检验
    - 如果某变量列联表不是有效的 2x2 表，卡方检验的解释需更谨慎
    """
    print_section("3. 二分类变量卡方检验")

    results = []

    for col in binary_feature_cols:
        contingency_table = pd.crosstab(df[col], df[target_col])

        # 若列联表维度不足，则跳过
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            continue

        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

        results.append({
            "variable": col,
            "chi2_statistic": round(chi2_stat, 4),
            "p_value": p_value
        })

    results_df = pd.DataFrame(results).sort_values(by="p_value", ascending=True).reset_index(drop=True)

    print("二分类变量卡方检验结果（按 p 值升序）：")
    print(results_df)

    return results_df


def random_forest_feature_importance(df: pd.DataFrame,
                                     target_col: str = "DEATH_EVENT",
                                     random_state: int = 42):
    """
    使用随机森林计算特征重要性。

    说明：
    - 这里只是做特征重要性分析
    - 随机森林不强依赖特征标准化，因此这里直接使用原始数值特征
    """
    print_section("4. 随机森林特征重要性")

    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=None
    )
    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("特征重要性结果（按重要性降序）：")
    print(importance_df)

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, save_path: str, top_n: int = 10):
    """
    绘制前10个最重要特征的条形图，并保存。

    修改点：
    - Top3 使用白色斜杠纹理
    - 背景网格线加深
    """
    print_section("5. 特征重要性条形图")

    # 取前 top_n 个变量
    top_df = importance_df.head(top_n).copy()
    top_df["rank_desc"] = np.arange(1, len(top_df) + 1)

    # 横向条形图排序（小在下，大在上）
    top_df = top_df.sort_values(by="importance", ascending=True).reset_index(drop=True)

    # 颜色设置
    color_map = {
        1: "#b22222",
        2: "#d89090",
        3: "#e5b5b5"
    }

    bar_colors = []
    bar_hatches = []
    edge_colors = []

    for _, row in top_df.iterrows():
        rank = row["rank_desc"]
        if rank in color_map:
            bar_colors.append(color_map[rank])
            bar_hatches.append("/")          # 斜杠
            edge_colors.append("white")      # 关键：白色斜杠通过边框实现
        else:
            bar_colors.append("#afc3d8")
            bar_hatches.append(None)
            edge_colors.append("#afc3d8")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(
        top_df["feature"],
        top_df["importance"],
        color=bar_colors,
        edgecolor=edge_colors,   # 控制斜杠颜色
        linewidth=1.0,
        height=0.65
    )

    # 设置斜杠纹理（Top3）
    for bar, hatch in zip(bars, bar_hatches):
        if hatch is not None:
            bar.set_hatch("/")  # 双斜杠更清晰一点（仍然属于“稀疏”风格）

    # 数值标注
    x_max = top_df["importance"].max()
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            ha="left",
            fontsize=10
        )

    ax.set_title("Top 10 Feature Importances from Random Forest", pad=10)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    # 去掉多余边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ✅ 加深背景虚线
    ax.grid(axis="x", linestyle="--", alpha=0.5, linewidth=0.8, color="#6b7280")
    ax.grid(axis="y", visible=False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"特征重要性图已保存到：{save_path}")



def summarize_top_risk_factors(t_test_df: pd.DataFrame,
                               chi2_df: pd.DataFrame,
                               importance_df: pd.DataFrame,
                               top_n: int = 3):
    """
    综合统计学检验结果与随机森林特征重要性，输出排名前三的危险因子。

    这里采用一个简单且易解释的综合策略：
    - 对连续变量：使用 T 检验 p 值排名
    - 对二分类变量：使用卡方检验 p 值排名
    - 对全部变量：使用随机森林重要性排名
    - 将“统计排名”和“模型排名”合并为综合分数，分数越小表示越重要
    """
    print_section("6. 综合输出排名前三的危险因子")

    # 统计检验结果合并成一个表
    stats_rows = []

    if t_test_df is not None and not t_test_df.empty:
        temp_t = t_test_df.copy()
        temp_t["test_type"] = "t_test"
        temp_t["stat_rank"] = temp_t["p_value"].rank(method="min", ascending=True)
        for _, row in temp_t.iterrows():
            stats_rows.append({
                "feature": row["variable"],
                "p_value": row["p_value"],
                "test_type": row["test_type"],
                "stat_rank": row["stat_rank"]
            })

    if chi2_df is not None and not chi2_df.empty:
        temp_c = chi2_df.copy()
        temp_c["test_type"] = "chi_square"
        temp_c["stat_rank"] = temp_c["p_value"].rank(method="min", ascending=True)
        for _, row in temp_c.iterrows():
            stats_rows.append({
                "feature": row["variable"],
                "p_value": row["p_value"],
                "test_type": row["test_type"],
                "stat_rank": row["stat_rank"]
            })

    stats_df = pd.DataFrame(stats_rows)

    # 随机森林排名
    rf_df = importance_df.copy()
    rf_df["rf_rank"] = rf_df["importance"].rank(method="min", ascending=False)

    # 合并统计检验排名和模型排名
    merged = pd.merge(
        rf_df[["feature", "importance", "rf_rank"]],
        stats_df[["feature", "p_value", "test_type", "stat_rank"]],
        on="feature",
        how="left"
    )

    # 综合分数：统计排名 + 随机森林排名
    # 若某特征没有统计检验结果，则给一个较大的默认值
    max_stat_rank = merged["stat_rank"].max() if merged["stat_rank"].notnull().any() else 10
    merged["stat_rank"] = merged["stat_rank"].fillna(max_stat_rank + 5)
    merged["combined_score"] = merged["rf_rank"] + merged["stat_rank"]

    merged = merged.sort_values(
        by=["combined_score", "importance"],
        ascending=[True, False]
    ).reset_index(drop=True)

    top_features = merged.head(top_n)

    print("综合排名前三的危险因子：\n")
    for idx, row in top_features.iterrows():
        feature = row["feature"]
        p_value = row["p_value"]
        test_type = row["test_type"]
        importance = row["importance"]

        print(f"第 {idx + 1} 名：{feature}")
        if pd.notnull(p_value):
            if test_type == "t_test":
                reason = f"该变量在连续变量 T 检验中具有较强统计显著性（p = {p_value:.4g}），且随机森林重要性较高（importance = {importance:.4f}）。"
            elif test_type == "chi_square":
                reason = f"该变量在卡方检验中具有较强统计显著性（p = {p_value:.4g}），且随机森林重要性较高（importance = {importance:.4f}）。"
            else:
                reason = f"该变量在统计检验和随机森林中都表现较突出（importance = {importance:.4f}）。"
        else:
            reason = f"该变量在随机森林中的重要性较高（importance = {importance:.4f}）。"


        print("原因：", reason)
        print("-" * 60)

    return merged, top_features


def save_results(t_test_df: pd.DataFrame,
                 chi2_df: pd.DataFrame,
                 importance_df: pd.DataFrame):
    """保存统计检验与特征重要性结果。"""
    print_section("7. 保存结果文件")

    t_test_path = os.path.join(RESULT_DIR, "continuous_t_test_results.csv")
    chi2_path = os.path.join(RESULT_DIR, "binary_chi_square_results.csv")
    importance_path = os.path.join(RESULT_DIR, "feature_importance_results.csv")

    t_test_df.to_csv(t_test_path, index=False, encoding="utf-8-sig")
    chi2_df.to_csv(chi2_path, index=False, encoding="utf-8-sig")
    importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")

    print(f"T 检验结果已保存到：{t_test_path}")
    print(f"卡方检验结果已保存到：{chi2_path}")
    print(f"特征重要性结果已保存到：{importance_path}")


def print_final_summary(t_test_df: pd.DataFrame,
                        chi2_df: pd.DataFrame,
                        importance_df: pd.DataFrame,
                        top_features: pd.DataFrame):
    """打印最终总结，方便直接写入论文初稿。"""
    print_section("8. 最终总结")

    # 统计检验中显著的变量（常用阈值 p < 0.05）
    sig_t = t_test_df[t_test_df["p_value"] < 0.05]["variable"].tolist() if not t_test_df.empty else []
    sig_c = chi2_df[chi2_df["p_value"] < 0.05]["variable"].tolist() if not chi2_df.empty else []

    top_rf = importance_df.head(5)["feature"].tolist() if not importance_df.empty else []
    top_3 = top_features["feature"].tolist() if top_features is not None and not top_features.empty else []

    print("统计检验中显著的连续变量（p < 0.05）：")
    print(sig_t if len(sig_t) > 0 else "无")

    print("\n统计检验中显著的二分类变量（p < 0.05）：")
    print(sig_c if len(sig_c) > 0 else "无")

    print("\n随机森林中重要性最高的前5个变量：")
    print(top_rf if len(top_rf) > 0 else "无")

    print("\n最终推荐重点关注的前三危险因子：")
    print(top_3 if len(top_3) > 0 else "无")

    print("\n简短总结：")
    print("1. 统计检验反映的是变量与死亡结局之间是否存在显著组间差异。")
    print("2. 随机森林特征重要性反映的是变量对分类模型划分的贡献程度。")
    print("3. 两者结合后，可初步识别出更值得重点关注的临床指标。")
    print("4. 这些结果更适合作为初步风险提示，不能直接视为严格因果结论。")


def main():
    ensure_directories()

    print_section("读取数据")
    df = load_data(FILE_PATH)
    print("数据前5行：")
    print(df.head())
    print("\n数据维度：", df.shape)

    target_col = "DEATH_EVENT"
    if target_col not in df.columns:
        raise ValueError(f"数据中不存在目标变量：{target_col}")
    print(f"\n已确认目标变量存在：{target_col}")

    print_section("变量类型区分")
    continuous_cols, binary_cols, binary_feature_cols = classify_variables(df, target_col=target_col)

    print("连续变量：")
    print(continuous_cols)

    print("\n二分类变量（含目标变量）：")
    print(binary_cols)

    print("\n用于卡方检验的二分类自变量：")
    print(binary_feature_cols)

    heatmap_path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    plot_correlation_heatmap(df, save_path=heatmap_path)

    t_test_df = t_test_continuous_variables(df, continuous_cols, target_col=target_col)

    chi2_df = chi_square_binary_variables(df, binary_feature_cols, target_col=target_col)

    importance_df = random_forest_feature_importance(df, target_col=target_col, random_state=RANDOM_STATE)

    importance_fig_path = os.path.join(FIG_DIR, "feature_importance.png")
    plot_feature_importance(importance_df, save_path=importance_fig_path, top_n=10)

    save_results(t_test_df, chi2_df, importance_df)

    merged_df, top_features = summarize_top_risk_factors(
        t_test_df=t_test_df,
        chi2_df=chi2_df,
        importance_df=importance_df,
        top_n=3
    )

    merged_df.to_csv(
        os.path.join(RESULT_DIR, "combined_risk_factor_ranking.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print_final_summary(t_test_df, chi2_df, importance_df, top_features)


if __name__ == "__main__":
    main()