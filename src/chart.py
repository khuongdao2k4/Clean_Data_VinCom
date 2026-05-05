from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("ggplot")
plt.rcParams["axes.unicode_minus"] = False


def find_project_root(start_dir: Path) -> Path:
    """
    Tim thu muc goc project dua tren thu muc hien tai cua file.

    Chart script hien dat tai `src/chart.py`, nhung van co the move ra root.
    Ham nay se tu dong tim thu muc chua `data/processed/`.
    """
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "data" / "processed").is_dir():
            return candidate
        if (candidate / "data").is_dir() and (candidate / "src").is_dir() and (candidate / "README.md").is_file():
            return candidate
    return start_dir


def require_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{dataset_name} thieu cot: {missing_str}")


def save_plot(output_dir: Path, filename: str) -> None:
    file_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Da luu: {file_path}")


def main() -> None:
    project_root = find_project_root(Path(__file__).resolve().parent)
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "output_charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_file = processed_dir / "cleaned_reviews.csv"
    labeled_file = processed_dir / "labeled_reviews.csv"

    if not cleaned_file.is_file():
        raise FileNotFoundError(
            f"Khong tim thay file: {cleaned_file}. Hay chay `python src/data_processing/clean_data.py` truoc."
        )
    if not labeled_file.is_file():
        raise FileNotFoundError(
            f"Khong tim thay file: {labeled_file}. Hay chay `python src/data_labeling/label_data.py` truoc."
        )

    cleaned_df = pd.read_csv(cleaned_file)
    labeled_df = pd.read_csv(labeled_file)

    require_columns(cleaned_df, ["mall_name", "rating", "review_date"], "cleaned_reviews.csv")
    require_columns(labeled_df, ["mall_name", "sentiment", "review_date"], "labeled_reviews.csv")

    cleaned_df["review_date"] = pd.to_datetime(cleaned_df["review_date"], errors="coerce")
    labeled_df["review_date"] = pd.to_datetime(labeled_df["review_date"], errors="coerce")

    cleaned_df["review_month"] = cleaned_df["review_date"].dt.to_period("M")
    labeled_df["review_month"] = labeled_df["review_date"].dt.to_period("M")

    # ============================================================
    # 1. PHAN BO DIEM DANH GIA
    # ============================================================
    rating_counts = cleaned_df["rating"].dropna().value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(rating_counts.index.astype(str), rating_counts.values)
    plt.title("Phan bo diem danh gia", fontsize=16, fontweight="bold")
    plt.xlabel("So sao", fontsize=12)
    plt.ylabel("So luong review", fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_plot(output_dir, "01_phan_bo_diem_danh_gia.png")

    # ============================================================
    # 2. DIEM TRUNG BINH THEO TUNG VINCOM
    # ============================================================
    avg_rating_by_mall = cleaned_df.groupby("mall_name")["rating"].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(avg_rating_by_mall.index, avg_rating_by_mall.values)
    plt.title("Diem trung binh theo tung Vincom", fontsize=16, fontweight="bold")
    plt.xlabel("Trung tam thuong mai", fontsize=12)
    plt.ylabel("Diem trung binh", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_plot(output_dir, "02_diem_trung_binh_theo_vincom.png")

    # ============================================================
    # 3. BOXPLOT RATING THEO TUNG VINCOM
    # ============================================================
    plt.figure(figsize=(12, 6))
    cleaned_df.boxplot(column="rating", by="mall_name", grid=False)
    plt.title("Phan bo rating theo tung Vincom", fontsize=16, fontweight="bold")
    plt.suptitle("")
    plt.xlabel("Trung tam thuong mai", fontsize=12)
    plt.ylabel("Rating", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    save_plot(output_dir, "03_boxplot_rating_theo_vincom.png")

    # ============================================================
    # 4. XU HUONG DIEM TRUNG BINH THEO THANG
    # ============================================================
    monthly_avg_rating = (
        cleaned_df.dropna(subset=["review_month"])
        .groupby("review_month")["rating"]
        .mean()
        .sort_index()
    )

    plt.figure(figsize=(12, 6))
    x_vals = monthly_avg_rating.index.astype(str)
    y_vals = monthly_avg_rating.values
    plt.plot(x_vals, y_vals, marker="o", linewidth=2)
    plt.title("Xu huong diem trung binh theo thang", fontsize=16, fontweight="bold")
    plt.xlabel("Thang", fontsize=12)
    plt.ylabel("Diem trung binh", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    for x, y in zip(x_vals, y_vals):
        plt.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    save_plot(output_dir, "04_xu_huong_diem_trung_binh_theo_thang.png")

    # ============================================================
    # 5. XU HUONG SO LUONG REVIEW THEO THANG
    # ============================================================
    monthly_review_count = (
        cleaned_df.dropna(subset=["review_month"])
        .groupby("review_month")
        .size()
        .sort_index()
    )

    plt.figure(figsize=(12, 6))
    bars = plt.bar(monthly_review_count.index.astype(str), monthly_review_count.values)
    plt.title("Xu huong so luong review theo thang", fontsize=16, fontweight="bold")
    plt.xlabel("Thang", fontsize=12)
    plt.ylabel("So luong review", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_plot(output_dir, "05_xu_huong_so_luong_review_theo_thang.png")

    # ============================================================
    # 6. TY LE SENTIMENT
    # ============================================================
    sentiment_counts = labeled_df["sentiment"].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    plt.title("Ty le sentiment trong tap du lieu", fontsize=16, fontweight="bold")

    save_plot(output_dir, "06_ty_le_sentiment.png")

    # ============================================================
    # 7. TY LE SENTIMENT THEO TUNG VINCOM
    # ============================================================
    sentiment_by_mall = pd.crosstab(
        labeled_df["mall_name"],
        labeled_df["sentiment"],
        normalize="index",
    )

    desired_order = ["negative", "neutral", "positive"]
    existing_cols = [col for col in desired_order if col in sentiment_by_mall.columns]
    sentiment_by_mall = sentiment_by_mall[existing_cols]

    sentiment_by_mall.plot(kind="bar", stacked=True, figsize=(12, 6), width=0.8)
    plt.title("Ty le sentiment theo tung Vincom", fontsize=16, fontweight="bold")
    plt.xlabel("Trung tam thuong mai", fontsize=12)
    plt.ylabel("Ty le", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Sentiment")
    plt.ylim(0, 1.05)

    save_plot(output_dir, "07_ty_le_sentiment_theo_vincom.png")

    print("\nHoan tat! Tat ca bieu do da duoc luu trong thu muc:")
    print(output_dir)


if __name__ == "__main__":
    main()
