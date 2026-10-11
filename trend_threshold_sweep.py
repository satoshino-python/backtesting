"""
ER のしきい値（と期間）を動かして、添付ルールの成績の変化を見る。
main_4H_trend_breakout_fast.py の売買ルール（ブレイク18本・SL1.0/TP1.5 ATR）はそのままで、フィルターの ER だけを変える。
しきい値 0 は「MA の並びだけ（ER を使わない）」。結果は trend_results/<日時>_er_threshold_sweep/trades_all.csv
"""
import contextlib, io, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import main_4H_fixedSL_multi as multi
import main_4H_trend_breakout_fast as m
from main_4H_fixedSL import (load_gmo_click_1min_data, resample_to_signal_bars, map_signals_to_1min,
                             get_trading_day_label, get_signal_bar_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for
from fast_engine import run_fast

PERIODS = (10, 20, 30)
THRESHOLDS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)


def run_pair(pair):
    multi.WARMUP_MONTHS = m.WARMUP_MONTHS
    cfg = replace(m.BASE_CFG, data_path=m.DATA_ROOT / pair, price_decimals=price_decimals_for(pair))
    rows = []
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, e = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df, spread = load_gmo_click_1min_data(cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
                                              month_range=zip_month_range(cfg), spread_period=(s, e),
                                              session_start_hour=cfg.session_start_hour)
        h4 = resample_to_signal_bars(df, cfg.session_start_hour, 4)
        d1 = resample_to_signal_bars(df, cfg.session_start_hour, 24)
        base = map_signals_to_1min(df, m.breakout_signals(h4, cfg.window, cfg.price_decimals), cfg.session_start_hour, 4)
        lab = get_signal_bar_label(df.index, cfg.session_start_hour, 24)
        day = get_trading_day_label(df.index, cfg.session_start_hour)
        inp = np.asarray((day >= s) & (day <= e))
        base = base.loc[inp]
        rate = spread / 2
        risk = cfg.cash * cfg.risk_pct
        comm = lambda size, price: round(abs(size) * price * rate, cfg.price_decimals)  # noqa: E731
        c = d1["Close"]
        ma_s, ma_m, ma_l = (c.rolling(p).mean() for p in m.MA_PERIODS)
        up_d, dn_d = (ma_s > ma_m) & (ma_m > ma_l), (ma_s < ma_m) & (ma_m < ma_l)
        for per in PERIODS:
            er_d = m.efficiency_ratio(c, per)
            f = pd.DataFrame({"up": up_d, "down": dn_d, "ER": er_d, "ok": ma_l.notna() & er_d.notna()}).shift(1).reindex(lab)
            ok = f["ok"].fillna(False).to_numpy(bool)[inp]
            er = f["ER"].to_numpy(float)[inp]
            up = f["up"].fillna(False).to_numpy(bool)[inp]; dn = f["down"].fillna(False).to_numpy(bool)[inp]
            for th in THRESHOLDS:
                strong = ok & (er >= th)
                d = base.copy(); d["AllowLong"], d["AllowShort"] = up & strong, dn & strong
                st = run_fast(d, cash=cfg.cash, commission=comm, margin=cfg.margin, sl_atr_multiplier=cfg.sl_atr_multiplier,
                              tp_atr_multiplier=cfg.tp_atr_multiplier, breakeven_trigger_r=None,
                              price_decimals=cfg.price_decimals, risk_pct=cfg.risk_pct, exit_mode="fixed")
                tr = st["_trades"].copy()
                tr["R"] = tr["PnL"] / risk; tr["Side"] = np.where(tr["Size"] > 0, "L", "S")
                tr.insert(0, "Threshold", th); tr.insert(0, "Period", per); tr.insert(0, "Pair", pair)
                rows.append(tr[["Pair", "Period", "Threshold", "EntryTime", "ExitTime", "Side", "R"]])
                rows[-1].attrs["on"] = (per, th, float((d["AllowLong"] | d["AllowShort"]).mean() * 100))
    on = pd.DataFrame([r.attrs["on"] for r in rows], columns=["Period", "Threshold", "OnPct"]); on.insert(0, "Pair", pair)
    return pd.concat(rows, ignore_index=True), on


if __name__ == "__main__":
    out = m.RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_er_threshold_sweep"
    out.mkdir(parents=True, exist_ok=True)
    tr, on = [], []
    with ProcessPoolExecutor(m.WORKERS) as ex:
        futs = {ex.submit(run_pair, p): p for p in m.PAIRS}
        for f in as_completed(futs):
            a, b = f.result(); tr.append(a); on.append(b); print("ok", futs[f], flush=True)
    pd.concat(tr, ignore_index=True).to_csv(out / "trades_all.csv", index=False, encoding="utf-8-sig")
    pd.concat(on, ignore_index=True).to_csv(out / "filter_on_share.csv", index=False)
    (out / "config.txt").write_text(f"periods={PERIODS}\nthresholds={THRESHOLDS}\nbase=main_4H_trend_breakout_fast.BASE_CFG\n")
    print("saved", out)
