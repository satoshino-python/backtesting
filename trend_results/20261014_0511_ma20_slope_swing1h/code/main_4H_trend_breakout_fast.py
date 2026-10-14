"""
トレンドフォロー戦略の検証（添付ルール trend_breakout_rules.md）。fast_engine で全ペアを並列に検証する。

- フィルター（日足）: 20MA > 75MA > 200MA かつ ER(20) >= 0.3 でロングのみ、逆の並びでショートのみ。
  4時間足の判定には「前日に確定した日足」の値を使う（NY17:00 区切りの日足を1本ずらす）。
- エントリー（4時間足）: 確定済みの直近 N(=18) 本の最高値/最安値への逆指値（判定中の足は含めない）。
- 決済: 4時間足 ATR(14, ワイルダー平滑化) × SL 1.0 / TP 1.5 の固定幅。建値ストップ・トレーリングなし。
- 枚数: 1R = 初期資金 × 2%（200固定・複利なし）を SL 幅で割る。
- スプレッド: データ全体の平均相対スプレッドの半分を売買の両方に課金（fast_engine 共通）。スワップは考慮しない。

フィルターの有無・パラメータ感度を同じ1分足で比較する。結果は trend_results/<日時>_<ラベル>/ に保存する。
"""
import contextlib
import io
import os
import json
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import main_4H_fixedSL_multi as multi
from main_4H_fixedSL import (Config, load_gmo_click_1min_data, resample_to_signal_bars,
                             map_signals_to_1min, get_trading_day_label, get_signal_bar_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for
from fast_engine import run_fast
from swap_costs import build_swap_model
from dow_swing_chart import make_chart

BASE_CFG = Config(start_date="2021-01-01", end_date="2025-12-31",
                  window=18, atr_period=14, sl_atr_multiplier=1.0, tp_atr_multiplier=1.5,
                  breakeven_trigger_r=None)
DATA_ROOT = Path("histData")
PAIRS = ("AUDJPY", "AUDUSD", "EURAUD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPAUD",
         "GBPCHF", "GBPJPY", "GBPUSD", "NZDJPY", "NZDUSD", "USDCHF", "USDJPY")
MA_PERIODS = (20, 75, 200)
ER_PERIOD = int(os.environ.get("ER_PERIOD", 20))   # 環境変数 ER_PERIOD で上書きできる（既定 20）
ER_THRESHOLD = 0.3
WARMUP_MONTHS = 12            # 200日MAの助走（日足200本 ≒ 10か月）
IS_END = pd.Timestamp("2023-12-31")
RESULTS_ROOT = Path("trend_results")
RUN_LABEL = "daily_ma_er_breakout18" if ER_PERIOD == 20 else f"daily_ma_er{ER_PERIOD}_breakout18"
WORKERS = 4

# 検証する設定。name: (フィルター使用, ブレイク本数, ER閾値, SL倍率, TP倍率)。先頭が添付ルールそのもの、2番目が比較用（フィルターなし）
VARIANTS = {"spec": (True, 18, 0.3, 1.0, 1.5), "nofilter": (False, 18, 0.3, 1.0, 1.5)}
for _er in (0.25, 0.35, 0.4):
    VARIANTS[f"er{_er}"] = (True, 18, _er, 1.0, 1.5)
for _n in (12, 24, 30):
    VARIANTS[f"brk{_n}"] = (True, _n, 0.3, 1.0, 1.5)
for _sl, _tp in ((0.8, 1.5), (1.5, 1.5), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (1.5, 3.0)):
    VARIANTS[f"sl{_sl}_tp{_tp}"] = (True, 18, 0.3, _sl, _tp)


def atr_wilder(high, low, close, period, decimals):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().round(decimals)


def efficiency_ratio(close, n):
    return (close - close.shift(n)).abs() / close.diff().abs().rolling(n).sum()


def breakout_signals(h4, n, decimals):
    """確定済み直近 n 本（判定中の足を含めない）の高値/安値と ATR を、判定足ごとに1本ずらして返す"""
    sig = pd.DataFrame({
        "SH": h4["High"].rolling(n).max(),
        "SL": h4["Low"].rolling(n).min(),
        "SHValid": 1.0, "SLValid": 1.0,
        "ATR": atr_wilder(h4["High"], h4["Low"], h4["Close"], 14, decimals),
    }, index=h4.index)
    return sig.shift(1)


def daily_filter(d1):
    """日足ごとの判定（その日足が確定した時点の値）。1=ロング許可 / -1=ショート許可 / 0=なし。ER も返す"""
    c = d1["Close"]
    ma_s, ma_m, ma_l = (c.rolling(p).mean() for p in MA_PERIODS)
    er = efficiency_ratio(c, ER_PERIOD)
    up, down = (ma_s > ma_m) & (ma_m > ma_l), (ma_s < ma_m) & (ma_m < ma_l)
    return pd.DataFrame({"up": up, "down": down, "ER": er})


def run_pair(pair, cfg, out_dir, make_charts):
    log = io.StringIO()
    t0 = time.perf_counter()
    multi.WARMUP_MONTHS = WARMUP_MONTHS
    with contextlib.redirect_stdout(log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_ts, end_ts = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df_1min, spread = load_gmo_click_1min_data(
            cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
            month_range=zip_month_range(cfg), spread_period=(start_ts, end_ts),
            session_start_hour=cfg.session_start_hour)
        h4 = resample_to_signal_bars(df_1min, cfg.session_start_hour, 4)
        d1 = resample_to_signal_bars(df_1min, cfg.session_start_hour, 24)
        dfl = daily_filter(d1).shift(1)               # 前日に確定した日足の判定
        label24 = get_signal_bar_label(df_1min.index, cfg.session_start_hour, 24)
        mapped = dfl.reindex(label24)
        er_1m = mapped["ER"].to_numpy(float)
        up_1m = mapped["up"].fillna(False).to_numpy(bool)
        down_1m = mapped["down"].fillna(False).to_numpy(bool)
        warm_1m = ~np.isnan(er_1m) & ~np.isnan(mapped["up"].astype(float).to_numpy())

        day = get_trading_day_label(df_1min.index, cfg.session_start_hour)
        in_period = np.asarray((day >= start_ts) & (day <= end_ts))
        rate = spread / 2
        risk = cfg.cash * cfg.risk_pct

        def comm(size, price):
            return round(abs(size) * price * rate, cfg.price_decimals)

        base_cols = df_1min
        rows, spec_stats, spec_state = [], None, None
        sig_cache = {}
        for name, (use_filter, n, er_th, sl, tp) in VARIANTS.items():
            if n not in sig_cache:
                sig_cache[n] = map_signals_to_1min(df_1min, breakout_signals(h4, n, cfg.price_decimals),
                                                   cfg.session_start_hour, 4)
            d = sig_cache[n].copy()
            strong = warm_1m & (er_1m >= er_th)
            al, as_ = (up_1m & strong, down_1m & strong) if use_filter else (np.ones(len(d), bool),) * 2
            d["AllowLong"], d["AllowShort"] = al, as_
            d = d.loc[in_period]
            st = run_fast(d, swap=build_swap_model(cfg.data_path, cfg.start_date, cfg.end_date, cfg.session_start_hour), cash=cfg.cash, commission=comm, margin=cfg.margin, sl_atr_multiplier=sl,
                          tp_atr_multiplier=tp, breakeven_trigger_r=None, price_decimals=cfg.price_decimals,
                          risk_pct=cfg.risk_pct, exit_mode="fixed")
            tr = st["_trades"].copy()
            tr["R"] = tr["PnL"] / risk
            tr["Side"] = np.where(tr["Size"] > 0, "L", "S")
            tr.insert(0, "Variant", name)
            tr.insert(0, "Pair", pair)
            rows.append(tr)
            on_share = float((al | as_)[in_period].mean() * 100) if use_filter else 100.0
            on_share_row = dict(Pair=pair, Variant=name, FilterOnPct=on_share)
            rows[-1].attrs["on"] = on_share_row
            if name == "spec":
                spec_stats = st
                spec_state = pd.Series(np.where(al & as_, 2, np.where(al, 1, np.where(as_, -1, 0))),
                                       index=df_1min.index)
        chart_path = None
        if make_charts:
            first_day, last_day = get_trading_day_label(df_1min.index[in_period], cfg.session_start_hour)[[0, -1]]
            chart_path = out_dir / f"dow_trade_chart_{pair}.html"
            make_chart(df_1min, chart_path,
                       f"{pair} {first_day:%Y-%m-%d} ~ {last_day:%Y-%m-%d} / 日足MA20>75>200+ER({ER_PERIOD})≥0.3 → 4H 18本ブレイク SL1.0/TP1.5 ATR",
                       first_day, last_day, trades=spec_stats["_trades"], weekly_trend=None, mode="fixed",
                       risk=risk, entry_window=cfg.window, entry_atr=14, price_decimals=cfg.price_decimals,
                       session_start_hour=cfg.session_start_hour, signal_hours=4,
                       filter_state=spec_state, filter_label=f"日足 MA20/75/200 並び + ER({ER_PERIOD})≥0.3（前日確定）",
                       daily_ma=MA_PERIODS, daily_er=(ER_PERIOD, ER_THRESHOLD))
    (out_dir / f"log_{pair}.txt").write_text(log.getvalue(), encoding="utf-8")
    on = pd.DataFrame([r.attrs["on"] for r in rows])
    return dict(pair=pair, trades=pd.concat(rows, ignore_index=True), on=on, chart=chart_path,
                spread=spread, seconds=time.perf_counter() - t0)


def main():
    out_dir = RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{RUN_LABEL}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "code").mkdir(exist_ok=True)
    here = Path(__file__).resolve().parent
    for f in (Path(__file__).name, "fast_engine.py", "main_4H_fixedSL.py"):
        shutil.copy2(here / f, out_dir / "code" / f)
    (out_dir / "config.json").write_text(json.dumps(
        dict(base_cfg={k: str(v) for k, v in asdict(BASE_CFG).items()}, pairs=PAIRS, variants=VARIANTS,
             ma=MA_PERIODS, er_period=ER_PERIOD, warmup_months=WARMUP_MONTHS), ensure_ascii=False, indent=2),
        encoding="utf-8")
    res = {}
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(run_pair, p, replace(BASE_CFG, data_path=DATA_ROOT / p,
                                                price_decimals=price_decimals_for(p)), out_dir, True): p
                for p in PAIRS}
        for f in as_completed(futs):
            p = futs[f]
            try:
                res[p] = f.result()
                print(f"✅ {p}: {res[p]['seconds']:.0f}秒", flush=True)
            except Exception as e:
                print(f"❌ {p}: {e!r}", flush=True)
    trades = pd.concat([res[p]["trades"] for p in PAIRS if p in res], ignore_index=True)
    trades.to_csv(out_dir / "trades_all.csv", index=False, encoding="utf-8-sig")
    pd.concat([r["on"] for r in res.values()]).to_csv(out_dir / "filter_on_share.csv", index=False)
    pd.Series({p: r["spread"] for p, r in res.items()}).to_csv(out_dir / "avg_relative_spread.csv")
    print("saved", out_dir)


if __name__ == "__main__":
    main()
