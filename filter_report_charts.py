"""
採用候補のフィルター（日足20日騰落が同方向 ＋ 4H移動平均が逆向きでない）で全ペアを再検証し、
レポート用の損益データとトレード付きダウ理論チャートを出力する。

- フィルターの定義と指標の計算は filter_rerun.py と同じ。比較用にフィルターなしも同時に実行する。
- 出力: filter_results/<日時>_d1roc_h4ma_charts/
  - trades_all.csv（filter, mode, Pair, EntryTime, ExitTime, R）
  - dow_trade_chart_<ペア>_<モード>.html（フィルターありのトレード）
"""
import contextlib
import io
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_fixedSL import (load_gmo_click_1min_data, resample_to_signal_bars, compute_signals,
                             map_signals_to_1min, get_trading_day_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for, compute_h1_swings
from main_4H_fixedSL_fast_multi import BASE_CFG
from fast_engine import run_fast
from dow_swing_chart import make_chart
from filter_rerun import FILTERS, MODES, H1_WINDOW, PAIRS, features_on_1min

TARGET = "d1roc_h4ma"
# 引数でフォルダを指定すると、そこに出力する（同じ実行のチャートを作り直すとき）
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("filter_results") / f"{datetime.now():%Y%m%d_%H%M}_{TARGET}_charts"
MODE_TEXT = {"fixed": "固定SL 1.5 / TP 2.5 ATR", "trail": "1Hスイング追従（前後5本・当初SL 1.5 ATR）"}


def run_pair(pair):
    rows = []
    log = io.StringIO()
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
        df_ext = map_signals_to_1min(df_1min, sig, 17, 4).join(compute_h1_swings(df_1min, H1_WINDOW, 17))
        day = get_trading_day_label(df_ext.index, 17)
        df = df_ext.loc[(day >= start_ts) & (day <= end_ts)].copy()
        first_day, last_day = get_trading_day_label(df.index, 17)[[0, -1]]
        feat = features_on_1min(pair, df.index)
        # チャートの背景用: 1分足（助走期間を含む全期間）でのフィルターの状態
        fa = features_on_1min(pair, df_1min.index)
        al = np.broadcast_to(np.asarray(FILTERS[TARGET][1](fa), bool), len(fa))
        as_ = np.broadcast_to(np.asarray(FILTERS[TARGET][2](fa), bool), len(fa))
        state = pd.Series(np.where(al & as_, 2, np.where(al, 1, np.where(as_, -1, 0))), index=df_1min.index)
        rate = spread / 2
        comm = lambda size, price: round(abs(size) * price * rate, cfg.price_decimals)  # noqa: E731
        risk = cfg.cash * cfg.risk_pct
        for fname in ("none", TARGET):
            _, fl, fs = FILTERS[fname]
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
                    rows.append(dict(filter=fname, mode=mode, Pair=pair, Side="L" if x.Size > 0 else "S",
                                     EntryTime=x.EntryTime, ExitTime=x.ExitTime, R=x.PnL / risk))
                if fname == TARGET:
                    make_chart(df_1min, OUT / f"dow_trade_chart_{pair}_{mode}.html",
                               f"{pair} {first_day:%Y-%m-%d} ~ {last_day:%Y-%m-%d} / {FILTERS[TARGET][0]} / {MODE_TEXT[mode]}",
                               first_day, last_day, trades=tr, weekly_trend=None, mode=MODES[mode]["exit_mode"],
                               risk=risk, entry_window=cfg.window, entry_atr=cfg.atr_period,
                               price_decimals=cfg.price_decimals, session_start_hour=17, signal_hours=4,
                               filter_state=state, filter_label=FILTERS[TARGET][0], ma_periods=(20, 50, 120))
    return pair, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(2) as ex:
        for pair, r in ex.map(run_pair, PAIRS):
            print(pair, len(r), flush=True)
            rows += r
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "trades_all.csv", index=False, encoding="utf-8-sig")
    print(t.groupby(["mode", "filter"]).R.agg(["count", "sum"]).round(2))
    print(f"結果: {OUT}")


if __name__ == "__main__":
    main()
