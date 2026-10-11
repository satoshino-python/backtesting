"""
ER の効果だけを取り出す検証。MA の並び（20>75>200 / 逆）だけでエントリーした全取引に、エントリー時（前日確定）の
日足 ER(10/20/30) を付ける。エントリー条件は固定なので、「ER が高い群と低い群で平均Rが違うか」を取引の入れ替わりなしに比べられる。
結果は trend_results/<日時>_er_split/trades_ma_only.csv
"""
import contextlib, io, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd

import main_4H_fixedSL_multi as multi
import main_4H_trend_breakout_fast as m
from main_4H_fixedSL import (load_gmo_click_1min_data, resample_to_signal_bars, map_signals_to_1min,
                             get_trading_day_label, get_signal_bar_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for
from fast_engine import run_fast

PERIODS = (10, 20, 30)


def run_pair(pair):
    multi.WARMUP_MONTHS = m.WARMUP_MONTHS
    cfg = replace(m.BASE_CFG, data_path=m.DATA_ROOT / pair, price_decimals=price_decimals_for(pair))
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
        c = d1["Close"]
        ma_s, ma_m, ma_l = (c.rolling(p).mean() for p in m.MA_PERIODS)
        f = pd.DataFrame({"up": (ma_s > ma_m) & (ma_m > ma_l), "down": (ma_s < ma_m) & (ma_m < ma_l), "ok": ma_l.notna()})
        for per in PERIODS:
            f[f"ER{per}"] = m.efficiency_ratio(c, per)
        f = f.shift(1).reindex(lab); f.index = df.index
        ok = f["ok"].fillna(False).to_numpy(bool)
        for per in PERIODS:                      # 3つの ER が全部そろうまでは取引しない（比較を同じ取引でそろえる）
            ok &= f[f"ER{per}"].notna().to_numpy()
        d = base.copy()
        d["AllowLong"], d["AllowShort"] = f["up"].fillna(False).to_numpy(bool) & ok, f["down"].fillna(False).to_numpy(bool) & ok
        d = d.loc[inp]
        rate = spread / 2; risk = cfg.cash * cfg.risk_pct
        comm = lambda size, price: round(abs(size) * price * rate, cfg.price_decimals)  # noqa: E731
        st = run_fast(d, cash=cfg.cash, commission=comm, margin=cfg.margin, sl_atr_multiplier=cfg.sl_atr_multiplier,
                      tp_atr_multiplier=cfg.tp_atr_multiplier, breakeven_trigger_r=None, price_decimals=cfg.price_decimals,
                      risk_pct=cfg.risk_pct, exit_mode="fixed")
        tr = st["_trades"].copy()
        tr["R"] = tr["PnL"] / risk; tr["Side"] = np.where(tr["Size"] > 0, "L", "S"); tr.insert(0, "Pair", pair)
        et = pd.DatetimeIndex(pd.to_datetime(tr["EntryTime"], utc=True)).tz_convert(df.index.tz)
        # エントリー時に有効だった（前日確定の）ER。エントリーは注文を置いた足の中で約定する
        pos = df.index.get_indexer(et)
        for per in PERIODS:
            tr[f"ER{per}"] = f[f"ER{per}"].to_numpy()[pos]
    return tr[["Pair", "EntryTime", "ExitTime", "Side", "R"] + [f"ER{p}" for p in PERIODS]]


if __name__ == "__main__":
    out = m.RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_er_split"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(m.WORKERS) as ex:
        futs = {ex.submit(run_pair, p): p for p in m.PAIRS}
        for fu in as_completed(futs):
            rows.append(fu.result()); print("ok", futs[fu], flush=True)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(out / "trades_ma_only.csv", index=False, encoding="utf-8-sig")
    print("saved", out, len(t), round(t.R.sum(), 1), t[["ER10", "ER20", "ER30"]].isna().sum().tolist())
