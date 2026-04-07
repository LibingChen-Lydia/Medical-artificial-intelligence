"""
心衰患者生存预测项目：数据初处理（Data Exploratory & Preprocessing）

数据集文件：
    heart_failure_clinical_records_dataset.csv

当前任务：
    1. 加载数据
    2. 基础信息查看
    3. 描述性统计分析
    4. 缺失值检查
    5. 异常值检测（IQR）
    6. 变量类型区分
    7. 可选标准化处理
    9. 输出简短总结

说明：
    - 医学数据中的极端值可能是真实临床现象，因此此阶段只识别和报告异常值，不直接删除
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# 为了让图表风格更规范，适合后续论文或报告使用
sns.set(style="whitegrid", font="DejaVu Sans")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


FILE_PATH = "heart_failure_clinical_records_dataset.csv"
TABLES_DIR = "tables"

MAIN_CONTINUOUS_CANDIDATES = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time"
]


def print_section(title: str):
    """打印分节标题，方便阅读输出结果。"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data(file_path: str) -> pd.DataFrame:
    """读取CSV数据。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据文件：{file_path}，请确认文件位于当前目录下。")
    df = pd.read_csv(file_path)
    return df


def classify_variables(df: pd.DataFrame):
    """
    自动区分变量类型：
    1. 二值变量：取值只有 2 个唯一值（常见如 0/1）
    2. 连续变量：数值型且不是二值变量

    说明：
    - 在这个数据集中，像 anaemia、diabetes、high_blood_pressure、sex、smoking、DEATH_EVENT是二分类变量。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    binary_cols = []
    continuous_cols = []

    for col in numeric_cols:
        unique_values = sorted(df[col].dropna().unique().tolist())
        if len(unique_values) == 2:
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    return binary_cols, continuous_cols


def basic_info(df: pd.DataFrame):
    """输出基础信息。"""
    print_section("1. 数据加载与基础信息")
    print("数据前5行：")
    print(df.head())

    print("\n数据维度（行数, 列数）：")
    print(df.shape)

    print("\n字段名称：")
    print(df.columns.tolist())

    print("\n数据类型：")
    print(df.dtypes)


def descriptive_statistics(df: pd.DataFrame, binary_cols: list, continuous_cols: list):
    """输出描述性统计分析。"""
    print_section("2. 描述性统计分析")

    if continuous_cols:
        print("连续变量描述性统计信息：")
        print(df[continuous_cols].describe().T)
    else:
        print("未识别到连续变量。")

    if binary_cols:
        print("\n二分类变量频数统计：")
        for col in binary_cols:
            print(f"\n变量：{col}")
            print(df[col].value_counts(dropna=False).sort_index())

    if "DEATH_EVENT" in df.columns:
        print("\n目标变量 DEATH_EVENT 的类别分布：")
        death_counts = df["DEATH_EVENT"].value_counts().sort_index()
        death_ratio = df["DEATH_EVENT"].value_counts(normalize=True).sort_index() * 100
        result = pd.DataFrame({
            "count": death_counts,
            "ratio(%)": death_ratio.round(2)
        })
        print(result)
    else:
        print("\n警告：数据中未找到目标变量 DEATH_EVENT。")

def generate_continuous_stats_latex(df: pd.DataFrame, continuous_cols: list,
                                    save_file: bool = True,
                                    file_name: str = "continuous_statistics_table.tex"):
    print_section("连续变量描述性统计的 LaTeX 表格")

    if not continuous_cols:
        print("未识别到连续变量，无法生成 LaTeX 表格。")
        return None, None

    os.makedirs(TABLES_DIR, exist_ok=True)

    stats_df = df[continuous_cols].describe().T[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]
    ].copy()

    stats_df = stats_df.rename(columns={
        "mean": "Mean",
        "std": "Std",
        "min": "Min",
        "25%": "Q1",
        "50%": "Median",
        "75%": "Q3",
        "max": "Max"
    }).round(2)

    stats_df = stats_df.reset_index().rename(columns={"index": "Variable"})

    latex_table = stats_df.to_latex(
        index=False,
        escape=False,
        caption="Descriptive statistics of continuous variables.",
        label="tab:continuous_stats",
        column_format="lrrrrrrr"
    )

    print("生成的 LaTeX 表格代码如下：\n")
    print(latex_table)

    if save_file:
        save_path = os.path.join(TABLES_DIR, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"\nLaTeX 表格已保存为：{save_path}")

    return stats_df, latex_table

def generate_categorical_stats_latex(df: pd.DataFrame, binary_cols: list,
                                     save_file=True,
                                     file_name="categorical_statistics_table.tex",
                                     exclude_target: bool = False):
    print_section("分类变量统计（LaTeX表）")

    os.makedirs(TABLES_DIR, exist_ok=True)

    cols_to_use = binary_cols.copy()
    if exclude_target and "DEATH_EVENT" in cols_to_use:
        cols_to_use.remove("DEATH_EVENT")

    rows = []

    for col in cols_to_use:
        counts = df[col].value_counts().sort_index()
        ratios = df[col].value_counts(normalize=True).sort_index() * 100

        for category in counts.index:
            rows.append({
                "Variable": col,
                "Category": category,
                "Count": int(counts[category]),
                "Percentage (%)": round(ratios[category], 2)
            })

    cat_df = pd.DataFrame(rows)

    latex_table = cat_df.to_latex(
        index=False,
        caption="Distribution of categorical variables.",
        label="tab:categorical_stats",
        column_format="lccc"
    )

    print(latex_table)

    if save_file:
        save_path = os.path.join(TABLES_DIR, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"\nLaTeX 表格已保存为：{save_path}")

    return cat_df, latex_table

def check_missing_values(df: pd.DataFrame):
    """检查缺失值情况。"""
    print_section("3. 缺失值检查")

    missing_count = df.isnull().sum()
    missing_ratio = df.isnull().mean() * 100

    missing_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_ratio(%)": missing_ratio.round(2)
    }).sort_values(by="missing_count", ascending=False)

    print("各列缺失值统计：")
    print(missing_df)

    total_missing = missing_count.sum()
    if total_missing == 0:
        print("\n结论：该数据集不存在缺失值。")
    else:
        print(f"\n结论：该数据集存在缺失值，总缺失数为 {total_missing}。")

    return missing_df, total_missing


def iqr_outlier_detection(df: pd.DataFrame, columns: list):
    """
    使用 IQR 方法检测异常值。
    返回每个变量的异常值个数、上下界等信息。
    """
    print_section("5. 异常值检测（IQR 方法）")

    outlier_summary = []

    for col in columns:
        if col not in df.columns:
            print(f"变量 {col} 不在数据集中，已跳过。")
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()
        outlier_ratio = outlier_count / len(df) * 100

        outlier_summary.append({
            "variable": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "outlier_count": int(outlier_count),
            "outlier_ratio(%)": round(outlier_ratio, 2)
        })

        print(f"{col}: 异常值数量 = {outlier_count}, 异常值比例 = {outlier_ratio:.2f}%")

    outlier_df = pd.DataFrame(outlier_summary).sort_values(by="outlier_count", ascending=False)

    print("\n异常值汇总表：")
    print(outlier_df)

    print("\n说明：")
    print("1. 这里使用 IQR 方法识别异常值，但暂时不删除。")
    print("2. 医学数据中的极端值可能是真实临床现象，而不一定是录入错误。")
    print("3. 后续建模时可以比较两种方案：")
    print("   - 方案A：保留异常值")
    print("   - 方案B：对极端值进行截尾（winsorize）或稳健处理")
    print("   然后比较模型效果与稳定性。")

    return outlier_df

def winsorize_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    对指定连续变量进行温和截尾（winsorize/clip）。
    这里只生成处理后的副本，不修改原始数据。
    """
    print_section("异常值温和处理（Winsorization）")

    df_winsorized = df.copy()

    for col in columns:
        if col not in df_winsorized.columns:
            continue

        q1 = df_winsorized[col].quantile(0.25)
        q3 = df_winsorized[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        before_outliers = ((df_winsorized[col] < lower_bound) | (df_winsorized[col] > upper_bound)).sum()

        df_winsorized[col] = df_winsorized[col].clip(lower=lower_bound, upper=upper_bound)

        after_outliers = ((df_winsorized[col] < lower_bound) | (df_winsorized[col] > upper_bound)).sum()

        print(f"{col}: 截尾前异常值数量 = {before_outliers}, 截尾后异常值数量 = {after_outliers}")

    return df_winsorized

'''
# platelets相对于其他变量量纲太大了，画箱形图不美观。进行其他变换会扭曲医学含义
def plot_boxplots(df: pd.DataFrame, columns: list, save_path: str = "figures/boxplot.png"):
    """绘制论文风格的连续变量箱线图（并保存图片）。"""
    print_section("连续变量箱线图")

    valid_cols = [col for col in columns if col in df.columns]

    if len(valid_cols) == 0:
        print("没有可绘制的连续变量。")
        return

    # ===== 确保保存路径存在 =====
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转换为 long-form
    plot_df = df[valid_cols].melt(var_name="Variable", value_name="Value")

    # 创建图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 箱线图
    sns.boxplot(
        data=plot_df,
        x="Variable",
        y="Value",
        ax=ax,
        width=0.55,
        showfliers=False,
        boxprops=dict(facecolor="#C7D7EB", edgecolor="#4C647A", linewidth=1.2),
        whiskerprops=dict(color="#4C647A", linewidth=1.1),
        capprops=dict(color="#4C647A", linewidth=1.1),
        medianprops=dict(color="#1F2D3A", linewidth=1.8)
    )

    # 散点
    sns.stripplot(
        data=plot_df,
        x="Variable",
        y="Value",
        ax=ax,
        color="#5B7C99",
        size=3,
        jitter=0.18,
        alpha=0.35
    )

    # 去边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.grid(axis="x", visible=False)

    ax.set_title("Distribution of Continuous Clinical Features", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Value")

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    # ===== 保存图片（关键新增）=====
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"图像已保存到：{save_path}")

    plt.show()
'''

def standardize_continuous_features(df: pd.DataFrame, continuous_cols: list):
    """
    只对连续变量做标准化，不对二值变量做标准化。
    返回标准化后的DataFrame和scaler对象。
    """
    print_section("8. 可选标准化处理")

    print("为什么有些模型需要标准化？")
    print("1. Logistic Regression、MLP 等模型通常依赖特征的数值尺度。")
    print("   如果不同特征量纲差异很大，模型训练可能变慢，或者某些特征会主导优化过程。")
    print("2. StandardScaler 会把连续变量转换为均值约为0、标准差约为1的形式，")
    print("   通常有助于提升这类模型的训练稳定性。")
    print("3. Random Forest、XGBoost、Decision Tree 等树模型基于特征分裂，")
    print("   一般不强依赖标准化，因此是否标准化对它们影响较小。")

    if not continuous_cols:
        print("\n未识别到连续变量，无法进行标准化。")
        return df.copy(), None

    df_scaled = df.copy()
    scaler = StandardScaler()

    # 只对连续变量进行标准化
    df_scaled[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    print("\n已完成连续变量标准化（二值变量保持不变）。")

    print("\n标准化前后的前5行对比（仅展示连续变量）：")
    print("\n【标准化前】")
    print(df[continuous_cols].head())

    print("\n【标准化后】")
    print(df_scaled[continuous_cols].head())

    return df_scaled, scaler


def final_summary(total_missing: int, outlier_df: pd.DataFrame):
    """输出简短总结。"""
    print_section("9. 数据初处理总结")

    # 1. 缺失值总结
    if total_missing == 0:
        missing_text = "数据集中未发现缺失值。"
    else:
        missing_text = f"数据集中存在缺失值，总缺失数为 {total_missing}。"

    # 2. 异常值较多的变量
    if outlier_df is not None and not outlier_df.empty:
        high_outlier_vars = outlier_df[outlier_df["outlier_count"] > 0]["variable"].tolist()
        if high_outlier_vars:
            outlier_text = f"检测到存在异常值的主要变量包括：{', '.join(high_outlier_vars)}。"
        else:
            outlier_text = "所检查的主要连续变量中未发现明显异常值。"
    else:
        outlier_text = "未生成异常值检测结果。"

    # 3. 标准化建议
    scaling_text = (
        "后续建模时，Logistic Regression、MLP、SVM 等对特征尺度较敏感的模型通常建议做标准化；"
        "而 Random Forest、Decision Tree 等树模型通常不强依赖标准化。"
    )

    print(missing_text)
    print(outlier_text)
    print(scaling_text)


def main():
    df = load_data(FILE_PATH)

    basic_info(df)

    print_section("变量类型区分")
    binary_cols, continuous_cols = classify_variables(df)

    print("自动识别的二分类变量：")
    print(binary_cols)

    print("\n自动识别的连续变量：")
    print(continuous_cols)

    print("\n说明：")
    print("像 anaemia、diabetes、high_blood_pressure、sex、smoking、DEATH_EVENT")
    print("虽然通常用 0/1 表示，但本质上是类别变量，因此归为二分类变量。")

    descriptive_statistics(df, binary_cols, continuous_cols)

    # 连续变量 LaTeX 表
    continuous_stats_df, continuous_stats_latex = generate_continuous_stats_latex(
        df,
        continuous_cols,  # 基线表中先不放派生变量
        save_file=True,
        file_name="continuous_statistics_table.tex"
    )

    # 分类变量 LaTeX 表
    categorical_stats_df, categorical_stats_latex = generate_categorical_stats_latex(
        df,
        binary_cols,
        save_file=True,
        file_name="categorical_statistics_table.tex",
        exclude_target=False   # 如果不想把 DEATH_EVENT 放进去，可改为 True
    )

    missing_df, total_missing = check_missing_values(df)

    main_continuous_vars = [col for col in MAIN_CONTINUOUS_CANDIDATES if col in df.columns]
    outlier_df = iqr_outlier_detection(df, main_continuous_vars)

    # 生成异常值温和处理版本（不覆盖原始数据）
    df_winsorized = winsorize_outliers(df, main_continuous_vars)
    df_winsorized.to_csv("heart_failure_clinical_records_dataset_winsorized.csv", index=False)

    df_scaled, scaler = standardize_continuous_features(df, continuous_cols)

    # 保存处理后数据
    df_scaled.to_csv("heart_failure_clinical_records_dataset_scaled.csv", index=False)

    final_summary(total_missing, outlier_df)


if __name__ == "__main__":
    main()
