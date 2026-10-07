"""
trend_filter_study.py で候補に挙がった方向フィルターを fast_engine に入れて、全ペアで再検証する。

- フィルターは「その方向の逆指値を置かない」形で入れる（fast_engine の AllowLong / AllowShort 列）。
  除外した取引の代わりに別の取引が入る影響も反映される。
- 指標は cache/h4_<ペア>.pkl（build_h4_cache.py。2020-01 から）で計算し、
  1分足の各バーには「そのバーの開始時刻までに終わった足」の値を付ける（ルックアヘッドなし）。
- 決済は fixed（SL 1.5 / TP 2.5 ATR）と swing_trail（1H 前後5本、当初SL 1.5 ATR）の両方。
- IS = 2021-2023、OOS = 2024-2025 に分けて集計する。

結果は filter_results/<日時>_rerun/ に保存する。
"""
import contextlib
import io
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_fixedSL import (load_gmo_click_1min_data, resample_to_signal_bars, compute_signals,
                             map_signals_to_1min, get_trading_day_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for, compute_h1_swings, max_drawdown
from main_4H_fixedSL_fast_multi import BASE_CFG
from fast_engine import run_fast
from trend_filter_study import build_features

PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "SP500", "USDCHF", "USDJPY")
MODES = {"fixed": dict(exit_mode="fixed"), "trail": dict(exit_mode="swing_trail", initial_sl_rule="atr")}
H1_WINDOW = 5
IS_END = pd.Timestamp("2023-12-31")

# 名前: (説明, 買いを許可する条件, 売りを許可する条件)。f は1分足に付けた指標の DataFrame
FILTERS = {
    "none": ("フィルターなし", lambda f: True, lambda f: True),
    "w1_dow_strict": ("週足ダウが同じ方向のときだけ",
                      lambda f: f.W1_DOW == 1, lambda f: f.W1_DOW == -1),
    "w1_dow_nocounter": ("週足ダウが逆方向のときは見送り",
                         lambda f: f.W1_DOW != -1, lambda f: f.W1_DOW != 1),
    "d1_roc20": ("日足の20日騰落が同じ方向のときだけ",
                 lambda f: f.D1_ROC20 > 0, lambda f: f.D1_ROC20 < 0),
    "h4_ma_nocounter": ("4H 移動平均(20/50/120)が逆向きに並んでいたら見送り",
                        lambda f: f.H4_MA_ORDER != -1, lambda f: f.H4_MA_ORDER != 1),
    "h4_ma_aligned": ("4H 移動平均が同じ方向に並んでいるときだけ",
                      lambda f: f.H4_MA_ORDER == 1, lambda f: f.H4_MA_ORDER == -1),
    "d1roc_h4ma": ("日足20日騰落が同方向 かつ 4H 移動平均が逆向きでない",
                   lambda f: (f.D1_ROC20 > 0) & (f.H4_MA_ORDER != -1),
                   lambda f: (f.D1_ROC20 < 0) & (f.H4_MA_ORDER != 1)),
    "d1_adx_lt25": ("日足 ADX(14) < 25 のときだけ（強トレンドを避ける）",
                    lambda f: f.D1_ADX14 < 25, lambda f: f.D1_ADX14 < 25),
    "d1_er20_dir": ("日足 ER(20) が取引方向にプラス",
                    lambda f: f.D1_ER20 > 0, lambda f: f.D1_ER20 < 0),
    "d1_er20_abs20": ("日足 |ER(20)| ≥ 0.2（方向は問わない）",
                      lambda f: f.D1_ER20.abs() >= 0.2, lambda f: f.D1_ER20.abs() >= 0.2),
    "h4_er18_dir20": ("4H ER(18) が取引方向に ≥ 0.2",
                      lambda f: f.H4_ER18 >= 0.2, lambda f: f.H4_ER18 <= -0.2),
    "w1_er13_abs30": ("週足 |ER(13)| ≥ 0.3（方向は問わない）",
                      lambda f: f.W1_ER13.abs() >= 0.3, lambda f: f.W1_ER13.abs() >= 0.3),
    "d1_adx_ge20": ("日足 ADX(14) ≥ 20 のときだけ（定番のトレンドフィルター）",
                    lambda f: f.D1_ADX14 >= 20, lambda f: f.D1_ADX14 >= 20),
}
NEEDED = ["W1_DOW", "D1_ROC20", "H4_MA_ORDER", "D1_ADX14", "D1_ER20", "H4_ER18", "W1_ER13"]


def features_on_1min(pair, index):
    t = pd.DataFrame({"t": index.tz_localize(None)})
    for f in build_features(pair):
        cols = [c for c in NEEDED if c in f.columns]
        if cols:
            t = pd.merge_asof(t, f[cols + ["end"]].sort_values("end"), left_on="t", right_on="end",
                              direction="backward").drop(columns="end")
    t.index = index
    return t


def run_pair(pair):
    log = io.StringIO()
    rows = []
    with contextlib.redirect_stdout(log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = replace(BASE_CFG, data_path=f"histData/{pair}", price_decimals=price_decimals_for(pair))
        start_ts, end_ts = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df_1min, spread = load_gmo_click_1min_data(cfg.data_path, price_side=cfg.price_side,
                                                   price_decimals=cfg.price_decimals,
                                                   month_range=zip_month_range(cfg),
                                                   spread_period=(start_ts, end_ts))
        sig = compute_signals(resample_to_signal_bars(df_1min, 17, 4), window=cfg.window,
                              atr_period=cfg.atr_period, price_decimals=cfg.price_decimals)
        df_ext = map_signals_to_1min(df_1min, sig, 17, 4)
        df_ext = df_ext.join(compute_h1_swings(df_1min, H1_WINDOW, 17))
        day = get_trading_day_label(df_ext.index, 17)
        df = df_ext.loc[(day >= start_ts) & (day <= end_ts)].copy()
        feat = features_on_1min(pair, df.index)
        rate = spread / 2
        comm = lambda size, price: round(abs(size) * price * rate, cfg.price_decimals)  # noqa: E731
        risk = cfg.cash * cfg.risk_pct
        for fname, (_, fl, fs) in FILTERS.items():
            d = df.copy()
            d["AllowLong"] = np.broadcast_to(np.asarray(fl(feat), bool), len(d))
            d["AllowShort"] = np.broadcast_to(np.asarray(fs(feat), bool), len(d))
            for mode, mp in MODES.items():
                params = dict(sl_atr_multiplier=cfg.sl_atr_multiplier, price_decimals=cfg.price_decimals,
                              risk_pct=cfg.risk_pct, **mp)
                if mode == "fixed":
                    params.update(tp_atr_multiplier=cfg.tp_atr_multiplier, breakeven_trigger_r=None)
                st = run_fast(d, cash=cfg.cash, commission=comm, margin=cfg.margin, **params)
                tr = st["_trades"]
                for _, x in tr.iterrows():
                    rows.append(dict(Pair=pair, filter=fname, mode=mode,
                                     EntryTime=x.EntryTime, R=x.PnL / risk, Side="L" if x.Size > 0 else "S"))
    return pair, rows


def stat(r):
    r = np.asarray(r, float)
    if len(r) == 0:
        return dict(n=0, R=0.0, avgR=np.nan, win=np.nan, PF=np.nan, maxDD=0.0)
    gl = -r[r < 0].sum()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max())
    return dict(n=len(r), R=r.sum(), avgR=r.mean(), win=(r > 0).mean() * 100,
                PF=r[r > 0].sum() / gl if gl > 0 else np.nan, maxDD=dd)


def main():
    out = Path("filter_results") / f"{datetime.now():%Y%m%d_%H%M}_rerun"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(2) as ex:
        for pair, r in ex.map(run_pair, PAIRS):
            print(pair, len(r), flush=True)
            rows += r
    t = pd.DataFrame(rows)
    t["EntryTime"] = pd.to_datetime(t["EntryTime"], utc=True).dt.tz_convert("America/New_York")
    t = t.sort_values("EntryTime")
    t["year"] = t.EntryTime.dt.year
    t["IS"] = t.EntryTime.dt.tz_localize(None) <= IS_END + pd.Timedelta(days=1)
    t.to_csv(out / "trades_all.csv", index=False, encoding="utf-8-sig")

    summ, yearly, pairs = [], [], []
    for (mode, fname), g in t.groupby(["mode", "filter"], sort=False):
        s = dict(mode=mode, filter=fname, desc=FILTERS[fname][0])
        for part, m in (("ALL", g.IS | ~g.IS), ("IS", g.IS), ("OOS", ~g.IS)):
            s.update({f"{part}_{k}": v for k, v in stat(g.loc[m, "R"]).items()})
        summ.append(s)
        for y, gy in g.groupby("year"):
            yearly.append(dict(mode=mode, filter=fname, year=int(y), R=gy.R.sum(), n=len(gy)))
        for p, gp in g.groupby("Pair"):
            pairs.append(dict(mode=mode, filter=fname, pair=p, R=gp.R.sum(), n=len(gp), avgR=gp.R.mean()))
    summ, yearly, pairs = pd.DataFrame(summ), pd.DataFrame(yearly), pd.DataFrame(pairs)
    summ.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(out / "by_pair.csv", index=False, encoding="utf-8-sig")
    (out / "filters.json").write_text(json.dumps({k: v[0] for k, v in FILTERS.items()}, ensure_ascii=False,
                                                 indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(summ[["mode", "filter", "ALL_n", "ALL_R", "ALL_avgR", "ALL_PF", "ALL_maxDD",
                "IS_n", "IS_R", "IS_avgR", "OOS_n", "OOS_R", "OOS_avgR"]].round(3).to_string(index=False))
    print(yearly.pivot_table(index=["mode", "filter"], columns="year", values="R").round(1).to_string())
    print(f"\n結果: {out}")


if __name__ == "__main__":
    main()
