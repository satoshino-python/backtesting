"""
FX版ミネルヴィニ戦略（docs/MINERVINI_FX_RULES.md）の検証を実行し、結果フォルダに保存する。

  python main_minervini_fx.py            # 基準（ルール通り）＋感度分析＋ペア別トレードチャート
  PARAM 上書きは minervini_fx.Params を dataclasses.replace で（このスクリプトの CFG は書き換えない）

エンジンは minervini_fx.py（日足判定・1分足約定）。スワップは swap_costs の FX 表を取引ごとに反映（Params.swap）。
結果は minervini_results/<日時>_<ラベル>/ に保存する（前回の結果を上書きしない）。
"""
import dataclasses
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import minervini_fx as mf

RESULTS_ROOT = Path("minervini_results")
RUN_LABEL = "FX15ペア_基準"
CFG = mf.Params()

# 感度分析: (名前, Params の上書き)。基準の周辺で成績が安定しているかを見る（最適値を探すためではない）
VARIANTS = [
    ("基準（ルール通り）", {}),
    ("ロングのみ", dict(sides=("long",))),
    ("ショートのみ", dict(sides=("short",))),
    ("同通貨・同方向 上限3本", dict(max_ccy_positions=3)),
    ("同通貨・同方向 上限2本", dict(max_ccy_positions=2)),
    ("同通貨・同方向 上限1本", dict(max_ccy_positions=1)),
    ("上位/下位 2×2", dict(top_n=2)),
    ("強さの差 ≥ 0.5", dict(min_gap=0.5)),
    ("強さの差 ≥ 1.0", dict(min_gap=1.0)),
    ("通貨強弱 42日", dict(ret_days=42)),
    ("通貨強弱 126日", dict(ret_days=126)),
    ("スイング 前後2日", dict(swing_window=2)),
    ("スイング 前後5日", dict(swing_window=5)),
    ("収縮 3回", dict(min_contractions=3)),
    ("200日線の傾き 3か月", dict(ma_slope_days=63)),
    ("200日線の傾き 5か月", dict(ma_slope_days=105)),
    ("52週位置 60%/40%", dict(long_pos=0.6)),
    ("52週位置 80%/20%", dict(long_pos=0.8)),
    ("損切り 1.5ATR", dict(sl_atr=1.5)),
    ("損切り 3ATR", dict(sl_atr=3.0)),
    ("手仕舞い 5日", dict(exit_days=5)),
    ("手仕舞い 20日", dict(exit_days=20)),
    ("10日安値/高値なし（損切りのみ）", dict(exit_days=None)),
]


def variant_row(name, trades):
    s = mf.overall(trades, "R")
    gross = trades["R_Gross"].to_numpy()
    nocost = ((trades["PnL_Gross"] + trades["SpreadUSD"]) / (CFG.cash * CFG.risk_pct)).to_numpy()
    return {"バリエーション": name, "トレード数": s["トレード数"], "合計R(スワップ後)": s["合計R"],
            "合計R(スワップ前)": gross.sum(), "合計R(コスト前)": nocost.sum(), "PF": s["PF"], "勝率[%]": s["勝率[%]"],
            "平均R": s["平均R"], "最大DD(R)": s["最大DD(R)"]}


def run_variants(data, cfg=CFG, variants=VARIANTS):
    rows = []
    for name, over in variants:
        p = dataclasses.replace(cfg, **over)
        tr = mf.run(data, p)
        rows.append(variant_row(name, tr) if len(tr) else {"バリエーション": name, "トレード数": 0})
    return pd.DataFrame(rows)


def bootstrap_total_r(r, n=10000, seed=0):
    """トレードの並べ替え（復元抽出）で合計Rの90%区間を出す。トレード数が少ないときの不確かさの目安"""
    rng = np.random.default_rng(seed)
    r = np.asarray(r, dtype=float)
    tot = rng.choice(r, size=(n, len(r)), replace=True).sum(axis=1)
    return float(np.percentile(tot, 5)), float(np.percentile(tot, 95))


def order_lines(data, sigs, pair):
    """その取引日に有効だった逆指値（前取引日の終値で確定したピボット。ロング＝SH、ショート＝SL）を、取引日で引ける表にする"""
    s = sigs[pair].iloc[:-1]
    return pd.DataFrame({"SH": np.where(s.Side == 1, s.Pivot, np.nan), "SL": np.where(s.Side == -1, s.Pivot, np.nan),
                         "ATR": s.Atr.to_numpy()}, index=data[pair].daily.index[1:])


def make_pair_chart(args):
    """（別プロセス用）1ペアのトレード付きチャートを作る"""
    pair, trades_csv, orders_csv, out_path, cfg_dict, title = args
    from dow_swing_chart import make_chart
    from main_4H_fixedSL import resample_to_signal_bars, signal_bar_trading_day
    from main_4H_fixedSL_multi import price_decimals_for
    cfg = mf.Params(**{**cfg_dict, "sides": tuple(cfg_dict["sides"])})
    d = mf.PairData(pair, cfg.end)
    df = pd.DataFrame({"Open": d.o, "High": d.h, "Low": d.l, "Close": d.c, "Volume": 0.0}, index=d.index_ny(slice(None)))
    bars = resample_to_signal_bars(df, 17, 4)
    day = pd.DatetimeIndex(signal_bar_trading_day(bars.index, 17))
    order = pd.read_csv(orders_csv, index_col=0, parse_dates=True)
    sig4 = order.reindex(day)
    sig4.index = bars.index
    sig4["SHValid"] = sig4["SH"].notna()
    sig4["SLValid"] = sig4["SL"].notna()
    make_chart(df, out_path, title, cfg.start, cfg.end, trades=pd.read_csv(trades_csv), mode="minervini",
               risk=cfg.cash * cfg.risk_pct, entry_window=cfg.swing_window, price_decimals=price_decimals_for(pair),
               entry_signals=sig4)
    return pair


def main():
    t0 = time.time()
    out = RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{RUN_LABEL}"
    out.mkdir(parents=True, exist_ok=True)
    data = mf.load_all(CFG.end)
    prepared = mf.prepare(data, CFG)
    trades = mf.run(data, CFG, prepared)
    trades.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    print(f"取引 {len(trades)} 件 / 合計 {trades['R'].sum():+.2f}R（スワップ前 {trades['R_Gross'].sum():+.2f}R）")

    variants = run_variants(data)
    variants.to_csv(out / "variants.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.unicode.east_asian_width", True):
        print(variants.round(2).to_string(index=False))

    (out / "config.json").write_text(json.dumps(dataclasses.asdict(CFG), ensure_ascii=False, indent=1, default=list),
                                     encoding="utf-8")
    strength = prepared[0]
    strength.to_csv(out / "currency_strength.csv", encoding="utf-8-sig")

    # ペア別チャート
    jobs = []
    for pair, g in trades.groupby("Pair"):
        csv = out / f"trades_{pair}.csv"
        g.to_csv(csv, index=False)
        ocsv = out / f"orders_{pair}.csv"
        order_lines(data, prepared[1], pair).to_csv(ocsv)
        jobs.append((pair, str(csv), str(ocsv), out / f"dow_trade_chart_{pair}_{CFG.start.replace('-', '')}-{CFG.end.replace('-', '')}_minervini.html",
                     json.loads(json.dumps(dataclasses.asdict(CFG), default=list)),
                     f"{pair} {CFG.start} ~ {CFG.end} / トレード: ミネルヴィニFX"))
    with ProcessPoolExecutor(3) as ex:
        for pair in ex.map(make_pair_chart, jobs):
            print("chart", pair)
    for j in jobs:
        Path(j[1]).unlink()
        Path(j[2]).unlink()
    for f in ("minervini_fx.py", "main_minervini_fx.py", "minervini_cache.py"):
        (out / "code").mkdir(exist_ok=True)
        shutil.copy(f, out / "code" / f)
    print(f"✅ {out}  ({time.time() - t0:.0f}秒)")
    return out, trades, variants


if __name__ == "__main__":
    main()
