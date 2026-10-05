"""
multi_results/ の下に保存された複数の実行結果（main_4H_fixedSL_multi.py の出力）を並べて比較する。

使い方:
    python compare_runs.py                 # multi_results/ 内の全ての実行を比較
    python compare_runs.py noBE BE1.0      # フォルダ名にこの文字列を含む実行だけを比較

出力: ターミナルに表を表示し、multi_results/comparison.csv に保存する。
"""
import sys
import json
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("multi_results")
# 総合成績（「合計」行）で比較する指標
TOTAL_COLS = ["トレード数", "勝率 [%]", "合計R", "平均R", "プロフィットファクター",
              "ペイオフレシオ", "最大連敗", "最大DD [R]", "収益率 [%]"]


def load_runs(filters):
    runs = []
    for d in sorted(p for p in RESULTS_ROOT.iterdir() if (p / "summary.csv").exists()):
        if filters and not any(f in d.name for f in filters):
            continue
        summary = pd.read_csv(d / "summary.csv", index_col=0, encoding="utf-8-sig")
        cfg_path = d / "config.json"
        cfg = {}
        if cfg_path.exists():
            info = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg = dict(info.get("base_cfg", {}))
            # base_cfg の外に記録している決済ルールの設定も比較対象にする
            # （記録が無い古い実行は従来の決済ルール = "fixed"）
            cfg["exit_mode"] = info.get("exit_mode", "fixed")
            cfg["h1_swing_window"] = info.get("h1_swing_window")
            cfg["initial_sl_rule"] = info.get("initial_sl_rule")
        runs.append((d.name, summary, cfg))
    return runs


def main():
    runs = load_runs(sys.argv[1:])
    if not runs:
        print(f"❌ 比較できる実行結果がありません（{RESULTS_ROOT}/<実行フォルダ>/summary.csv）")
        return

    # 1) 総合成績
    total = pd.DataFrame({name: s.loc["合計", TOTAL_COLS] for name, s, _ in runs}).T
    # 2) 通貨ペア別の合計R
    pair_r = pd.DataFrame({name: s.drop(index="合計")["合計R"] for name, s, _ in runs}).T
    # 3) 実行ごとに違う設定だけを抜き出す（何を変えた実行かが一目で分かるように）
    cfgs = pd.DataFrame({name: c for name, _, c in runs}).T
    # None は astype(str) では文字列にならず nunique で数えられないため repr で比較する
    diff_cfg = cfgs.loc[:, cfgs.map(repr).nunique() > 1] if len(cfgs.columns) else cfgs

    with pd.option_context("display.unicode.east_asian_width", True,
                           "display.width", 250, "display.max_columns", None):
        if not diff_cfg.empty:
            print("\n================ 実行ごとに異なる設定 ================")
            print(diff_cfg.to_string(na_rep="（記録なし）"))
        print("\n================ 総合成績 ================")
        print(total.to_string(float_format=lambda v: f"{v:,.2f}"))
        print("\n================ 通貨ペア別 合計R ================")
        print(pair_r.to_string(float_format=lambda v: f"{v:,.2f}"))

    out = pd.concat({"設定": diff_cfg, "総合": total, "ペア別合計R": pair_r}, axis=1)
    out.to_csv(RESULTS_ROOT / "comparison.csv", encoding="utf-8-sig")
    print(f"\n💾 {RESULTS_ROOT / 'comparison.csv'} に保存しました")


if __name__ == "__main__":
    main()
