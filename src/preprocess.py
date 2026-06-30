"""将原始分钟级电力数据与月度气象数据处理为日级训练/测试数据。"""

from __future__ import annotations

import argparse
import gzip
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


POWER_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

WEATHER_COLUMNS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]


def load_power(raw_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(raw_zip) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        with zf.open(name) as f:
            df = pd.read_csv(
                f,
                sep=";",
                na_values=["?"],
                low_memory=False,
            )
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df.insert(0, "DateTime", dt)
    for col in POWER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["DateTime"])


def load_weather(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            weather = pd.read_csv(f, sep=";", low_memory=False)
    else:
        weather = pd.read_csv(path, sep=";", low_memory=False)

    # Meteo-France 的字段名可能随资源略有变化。优先使用 AAAAMM 或 DATE 类字段。
    date_col = next((c for c in ["AAAAMM", "DATE", "date"] if c in weather.columns), None)
    if date_col is None:
        raise ValueError(f"气象数据缺少月份字段，实际字段：{weather.columns.tolist()[:20]}")

    out = pd.DataFrame()
    raw_date = weather[date_col].astype(str).str.replace(r"\D", "", regex=True).str[:6]
    out["Month"] = pd.to_datetime(raw_date + "01", format="%Y%m%d", errors="coerce")

    for col in WEATHER_COLUMNS:
        if col in weather.columns:
            out[col] = pd.to_numeric(weather[col], errors="coerce")
        else:
            out[col] = np.nan

    out = out.dropna(subset=["Month"])
    out = out.groupby("Month", as_index=False)[WEATHER_COLUMNS].mean()
    return out


def aggregate_daily(power: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    power["Date"] = power["DateTime"].dt.date
    daily = power.groupby("Date").agg(
        {
            "Global_active_power": "sum",
            "Global_reactive_power": "sum",
            "Sub_metering_1": "sum",
            "Sub_metering_2": "sum",
            "Sub_metering_3": "sum",
            "Voltage": "mean",
            "Global_intensity": "mean",
        }
    )

    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D")
    numeric_cols = daily.columns.tolist()
    daily[numeric_cols] = daily[numeric_cols].interpolate(limit_direction="both")

    daily["Sub_metering_remainder"] = (
        daily["Global_active_power"] * 1000 / 60
        - daily["Sub_metering_1"]
        - daily["Sub_metering_2"]
        - daily["Sub_metering_3"]
    )

    if weather is not None:
        daily = daily.reset_index(names="Date")
        daily["Month"] = daily["Date"].values.astype("datetime64[M]")
        daily = daily.merge(weather, on="Month", how="left").drop(columns=["Month"])
        daily[WEATHER_COLUMNS] = daily[WEATHER_COLUMNS].ffill().bfill()
    else:
        daily = daily.reset_index(names="Date")
        for col in WEATHER_COLUMNS:
            daily[col] = 0.0

    ordered_cols = [
        "Date",
        "Global_active_power",
        "Global_reactive_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
        "Sub_metering_remainder",
        "Voltage",
        "Global_intensity",
        *WEATHER_COLUMNS,
    ]
    return daily[ordered_cols]


def split_train_test(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily.sort_values("Date")
    train = daily[daily["Date"] < "2009-01-01"].copy()
    test = daily[daily["Date"] >= "2009-01-01"].copy()
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--use-reference-processed", action="store_true")
    parser.add_argument(
        "--reference-dir",
        default="../参考/参考内容3/machine-learning-homework/data/data",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.use_reference_processed:
        ref = Path(args.reference_dir)
        shutil.copy2(ref / "train.csv", processed_dir / "train.csv")
        shutil.copy2(ref / "test.csv", processed_dir / "test.csv")
        print("已复制参考日级数据。注意：正式报告仍以考核 PDF 指定的两个网站作为数据来源。")
        return

    raw_dir = Path(args.raw_dir)
    power = load_power(raw_dir / "individual_household_power_consumption.zip")
    weather_files = sorted(raw_dir.glob("meteo_mensq_*_1950_2024.csv*"))
    weather = load_weather(weather_files[0]) if weather_files else None

    daily = aggregate_daily(power, weather)
    train, test = split_train_test(daily)
    train.to_csv(processed_dir / "train.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)
    print(f"训练集：{train.shape} -> {processed_dir / 'train.csv'}")
    print(f"测试集：{test.shape} -> {processed_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
