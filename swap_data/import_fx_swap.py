"""
Googleスプレッドシート「GMO FXスワップ」をCSVでダウンロードしたファイルから、swap_data/fx_swap.csv を作り直す。

    python swap_data/import_fx_swap.py <ダウンロードしたCSV> [出力先 (既定: swap_data/fx_swap.csv)]

- 列: 通貨ペア, 取引日, 売Swap (円), 買Swap (円), 付与日数（単位は1万通貨あたりの円）
- 付与日数が 0 の日（空欄を含む）は省く
- 「64$0.41」「-64-$0.41」のように円とドルが続いて入っている値は、円の部分だけを使う。「1,116」の桁区切りも読める
- 出力: pair, date, sell, buy, days（日付順）
"""
import re
import sys
from pathlib import Path

import pandas as pd


def parse_yen(value):
    if pd.isna(value):
        return 0.0
    m = re.match(r"^\s*(-?[\d,]+(?:\.\d+)?)", str(value))
    if not m:
        raise ValueError(f"円の金額として読めません: {value!r}")
    return float(m.group(1).replace(",", ""))


def convert(src):
    raw = pd.read_csv(src, dtype=str, encoding="utf-8-sig")
    out = pd.DataFrame({
        "pair": raw["通貨ペア"].str.strip(),
        "date": pd.to_datetime(raw["取引日"], format="%Y/%m/%d"),
        "sell": raw["売Swap (円)"].map(parse_yen),
        "buy": raw["買Swap (円)"].map(parse_yen),
        "days": raw["付与日数"].astype(int),
    })
    if out.duplicated(["pair", "date"]).any():
        raise ValueError("同じ通貨ペア・日付の行が複数あります")
    out = out[out["days"] > 0].sort_values(["pair", "date"]).reset_index(drop=True)
    out[["sell", "buy"]] = out[["sell", "buy"]].round().astype(int)
    return out


if __name__ == "__main__":
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "fx_swap.csv"
    table = convert(sys.argv[1])
    table.to_csv(dst, index=False)
    print(f"{dst}: {len(table)} 行")
    print(table.groupby("pair")["date"].agg(["min", "max", "count"]).to_string())
