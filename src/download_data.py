"""下载课程项目所需的原始数据。

数据来源与考核 PDF 一致：
1. UCI 家庭电力分钟级数据；
2. data.gouv.fr / Meteo-France 月度基础气象数据。
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


UCI_POWER_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
METEO_API_URL = (
    "https://www.data.gouv.fr/api/1/datasets/"
    "donnees-climatologiques-de-base-mensuelles/"
)


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"已存在，跳过下载：{target}")
        return
    print(f"下载：{url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())
    print(f"保存：{target} ({target.stat().st_size} bytes)")


def fetch_meteo_resources(department: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(METEO_API_URL, timeout=60) as response:
        dataset = json.load(response)

    pattern = re.compile(rf"MENS_departement_{re.escape(department)}_periode_1950-2024")
    resources = []
    for resource in dataset.get("resources", []):
        title = resource.get("title", "")
        url = resource.get("url", "")
        fmt = resource.get("format", "")
        if pattern.search(title) and url and fmt in {"csv.gz", "csv"}:
            resources.append({"title": title, "url": url, "format": fmt})
    return resources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--department",
        default="92",
        help="法国省份编号。UCI 数据来自巴黎近郊家庭，默认取 Hauts-de-Seine(92) 的月度气象数据。",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    download(UCI_POWER_URL, raw_dir / "individual_household_power_consumption.zip")

    resources = fetch_meteo_resources(args.department)
    if not resources:
        raise SystemExit(f"未找到 department={args.department} 的 Meteo-France 月度气象资源")

    meta_path = raw_dir / "meteo_resources.json"
    meta_path.write_text(json.dumps(resources, ensure_ascii=False, indent=2), encoding="utf-8")
    for resource in resources:
        suffix = ".csv.gz" if resource["format"] == "csv.gz" else ".csv"
        target = raw_dir / f"meteo_mensq_{args.department}_1950_2024{suffix}"
        download(resource["url"], target)


if __name__ == "__main__":
    main()
