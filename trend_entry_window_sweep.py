"""
スイングハイ・ローのエントリーで、足の長さと本数を動かして成績の変化を見る（1H: 6〜48本 / 4H: 3〜18本）。
フィルターは 日足MA20>75>200 の並び + ER(20)≥0.3（+ 比較用に「フィルターなし」「MAの並びだけ」「+ 20MA傾き1日」）。
SL/TP は 4時間足 ATR(14)（ワイルダー）× 1.0 / 1.5 で共通。結果は trend_results/<日時>_entry_window_sweep/
"""
import contextlib, io, os, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
import numpy as np, pandas as pd
import main_4H_fixedSL_multi as multi
import main_4H_trend_breakout_fast as m
from main_4H_fixedSL import (load_gmo_click_1min_data, resample_to_signal_bars, compute_signals, map_signals_to_1min,
                             get_trading_day_label, get_signal_bar_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for
from fast_engine import run_fast

CONFIGS = [(1, w) for w in (6, 12, 18, 24, 36, 48)] + [(4, w) for w in (3, 6, 9, 12, 18)]   # (足の時間, 前後の本数)
# 環境変数 CONFIGS="2:3,2:6,3:2" のように指定すると、その組み合わせだけを検証する（例: 2時間足3本、2時間足6本、3時間足2本）
if os.environ.get("CONFIGS"):
    CONFIGS = [tuple(int(x) for x in c.split(":")) for c in os.environ["CONFIGS"].split(",")]
LABEL = os.environ.get("RUN_LABEL", "entry_window_sweep")
VARIANTS = ("nofilter", "ma_only", "spec", "spec_slope1")


def one(pair):
    multi.WARMUP_MONTHS = m.WARMUP_MONTHS
    cfg = replace(m.BASE_CFG, data_path=m.DATA_ROOT / pair, price_decimals=price_decimals_for(pair))
    out, ons = [], []
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, e = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df, spread = load_gmo_click_1min_data(cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
                                              month_range=zip_month_range(cfg), spread_period=(s, e), session_start_hour=cfg.session_start_hour)
        h = {hh: resample_to_signal_bars(df, 17, hh) for hh in sorted({c[0] for c in CONFIGS} | {4})}
        d1 = resample_to_signal_bars(df, 17, 24)
        atr4 = m.atr_wilder(h[4]["High"], h[4]["Low"], h[4]["Close"], 14, cfg.price_decimals).shift(1)
        lab = get_signal_bar_label(df.index, 17, 24); day = get_trading_day_label(df.index, 17); inp = np.asarray((day >= s) & (day <= e))
        rate = spread / 2; risk = cfg.cash * cfg.risk_pct
        comm = lambda size, price: round(abs(size) * price * rate, cfg.price_decimals)  # noqa: E731
        c = d1["Close"]; ma_s, ma_m, ma_l = (c.rolling(p).mean() for p in m.MA_PERIODS); er = m.efficiency_ratio(c, 20)
        f = pd.DataFrame({"up": (ma_s > ma_m) & (ma_m > ma_l), "dn": (ma_s < ma_m) & (ma_m < ma_l), "ok": ma_l.notna() & er.notna() & ma_s.shift(1).notna(),
                          "er": er >= 0.3, "su": ma_s > ma_s.shift(1), "sd": ma_s < ma_s.shift(1)}).shift(1).reindex(lab)
        g = {k: f[k].fillna(False).to_numpy(bool) for k in f.columns}
        allow = {"nofilter": (np.ones(len(df), bool),) * 2,
                 "ma_only": (g["up"] & g["ok"], g["dn"] & g["ok"]),
                 "spec": (g["up"] & g["ok"] & g["er"], g["dn"] & g["ok"] & g["er"]),
                 "spec_slope1": (g["up"] & g["ok"] & g["er"] & g["su"], g["dn"] & g["ok"] & g["er"] & g["sd"])}
        for hours, w in CONFIGS:
            sig = compute_signals(h[hours], window=w, atr_period=14, price_decimals=cfg.price_decimals)
            sig["ATR"] = (atr4.reindex(get_signal_bar_label(h[hours].index, 17, 4)).to_numpy() if hours != 4 else atr4)
            base = map_signals_to_1min(df, sig, 17, hours).loc[inp]
            for v in VARIANTS:
                d = base.copy(); d["AllowLong"], d["AllowShort"] = allow[v][0][inp], allow[v][1][inp]
                st = run_fast(d, cash=cfg.cash, commission=comm, margin=cfg.margin, sl_atr_multiplier=cfg.sl_atr_multiplier,
                              tp_atr_multiplier=cfg.tp_atr_multiplier, breakeven_trigger_r=None, price_decimals=cfg.price_decimals,
                              risk_pct=cfg.risk_pct, exit_mode="fixed")
                tr = st["_trades"].copy(); tr["R"] = tr["PnL"] / risk; tr["Side"] = np.where(tr["Size"] > 0, "L", "S")
                tr.insert(0, "Window", w); tr.insert(0, "Hours", hours); tr.insert(0, "Variant", v); tr.insert(0, "Pair", pair)
                out.append(tr[["Pair", "Variant", "Hours", "Window", "EntryTime", "ExitTime", "Side", "R"]])
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    out = m.RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{LABEL}"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(m.WORKERS) as ex:
        futs = {ex.submit(one, p): p for p in m.PAIRS}
        for fu in as_completed(futs):
            rows.append(fu.result()); print("ok", futs[fu], flush=True)
    pd.concat(rows, ignore_index=True).to_csv(out / "trades_all.csv", index=False, encoding="utf-8-sig")
    print("saved", out)
