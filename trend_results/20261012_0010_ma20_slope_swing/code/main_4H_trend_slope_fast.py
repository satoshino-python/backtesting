"""
トレンドブレイクアウト検証に「日足20MAの傾き」の条件を追加した版（main_4H_trend_breakout_fast.py の拡張）。

追加条件: ロングは日足 20MA が上向き（MA20[前日] > MA20[前日 - k日]）、ショートは下向き。前日確定の日足で判定する。
他のルール（MA20>75>200 の並び、ER(20)≥0.3、4H 18本ブレイク、SL1.0/TP1.5 ATR）は main_4H_trend_breakout_fast.py と同じ。
結果は trend_results/<日時>_ma20_slope/ に保存する。
"""
import contextlib, io, json, os, shutil, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import main_4H_fixedSL_multi as multi
import main_4H_trend_breakout_fast as m
from main_4H_fixedSL import (load_gmo_click_1min_data, resample_to_signal_bars, map_signals_to_1min,
                             get_trading_day_label, get_signal_bar_label, compute_signals)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for
from fast_engine import run_fast
from dow_swing_chart import make_chart

# エントリー方式（環境変数 ENTRY_MODE で切り替え）
#   donchian: 確定済み直近18本の最高値/最安値（添付ルール。既定）
#   swing   : 既存戦略と同じ Swing High/Low（前後18本で確定。一度抜かれたラインは次のスイングが確定するまで無効）
ENTRY_MODE = os.environ.get("ENTRY_MODE", "donchian")
RUN_LABEL = "ma20_slope" if ENTRY_MODE == "donchian" else f"ma20_slope_{ENTRY_MODE}"
# name: (ER期間, ERしきい値, 傾きの日数k（0=条件なし）, ER を使うか)。SL1.0 / TP1.5 / ブレイク18本は共通
VARIANTS = {
    "nofilter":      None,
    "spec":          (20, 0.3, 0, True),      # 添付ルール（ER20 ≥ 0.3）
    "spec_slope1":   (20, 0.3, 1, True),      # + 20MAが1日前より上/下
    "spec_slope3":   (20, 0.3, 3, True),
    "spec_slope5":   (20, 0.3, 5, True),
    "er10":          (10, 0.3, 0, True),
    "er10_slope1":   (10, 0.3, 1, True),
    "ma_only":       (20, 0.0, 0, False),     # MAの並びだけ
    "ma_slope1":     (20, 0.0, 1, False),     # MAの並び + 傾きだけ（ERなし）
    "ma_slope3":     (20, 0.0, 3, False),
}
for _th in (0.2, 0.25, 0.35, 0.4):
    VARIANTS[f"slope1_er{_th}"] = (20, _th, 1, True)
CHART_VARIANT = "spec_slope1"


def run_pair(pair, cfg, out_dir):
    multi.WARMUP_MONTHS = m.WARMUP_MONTHS
    log = io.StringIO(); t0 = time.perf_counter()
    with contextlib.redirect_stdout(log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, e = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df, spread = load_gmo_click_1min_data(cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
                                              month_range=zip_month_range(cfg), spread_period=(s, e),
                                              session_start_hour=cfg.session_start_hour)
        h4 = resample_to_signal_bars(df, cfg.session_start_hour, 4)
        d1 = resample_to_signal_bars(df, cfg.session_start_hour, 24)
        if ENTRY_MODE == "swing":
            sig = compute_signals(h4, window=cfg.window, atr_period=14, price_decimals=cfg.price_decimals)
            sig["ATR"] = m.atr_wilder(h4["High"], h4["Low"], h4["Close"], 14, cfg.price_decimals).shift(1)  # ATR は他の版と同じワイルダー平滑
        else:
            sig = m.breakout_signals(h4, cfg.window, cfg.price_decimals)
        base = map_signals_to_1min(df, sig, cfg.session_start_hour, 4)
        lab = get_signal_bar_label(df.index, cfg.session_start_hour, 24)
        day = get_trading_day_label(df.index, cfg.session_start_hour)
        inp = np.asarray((day >= s) & (day <= e))
        base = base.loc[inp]
        rate = spread / 2; risk = cfg.cash * cfg.risk_pct
        comm = lambda size, price: round(abs(size) * price * rate, cfg.price_decimals)  # noqa: E731
        c = d1["Close"]
        ma_s, ma_m, ma_l = (c.rolling(p).mean() for p in m.MA_PERIODS)
        up_d, dn_d = (ma_s > ma_m) & (ma_m > ma_l), (ma_s < ma_m) & (ma_m < ma_l)

        def daily_state(per, k):
            er = m.efficiency_ratio(c, per)
            f = pd.DataFrame({"up": up_d, "down": dn_d, "ER": er,
                              "sup": (ma_s > ma_s.shift(k)) if k else True, "sdn": (ma_s < ma_s.shift(k)) if k else True,
                              "ok": ma_l.notna() & er.notna() & (ma_s.shift(k).notna() if k else True)}).shift(1).reindex(lab)
            return f

        rows, state_chart, ons = [], None, []
        for name, v in VARIANTS.items():
            if v is None:
                al = as_ = np.ones(len(base), bool)
            else:
                per, th, k, use_er = v
                f = daily_state(per, k)
                ok = f["ok"].fillna(False).to_numpy(bool)
                strong = ok & ((f["ER"].to_numpy(float) >= th) if use_er else True)
                full_l = f["up"].fillna(False).to_numpy(bool) & f["sup"].fillna(False).to_numpy(bool) & strong
                full_s = f["down"].fillna(False).to_numpy(bool) & f["sdn"].fillna(False).to_numpy(bool) & strong
                al, as_ = full_l[inp], full_s[inp]
                if name == CHART_VARIANT:
                    state_chart = pd.Series(np.where(full_l & full_s, 2, np.where(full_l, 1, np.where(full_s, -1, 0))), index=df.index)
            d = base.copy(); d["AllowLong"], d["AllowShort"] = al, as_
            st = run_fast(d, cash=cfg.cash, commission=comm, margin=cfg.margin, sl_atr_multiplier=cfg.sl_atr_multiplier,
                          tp_atr_multiplier=cfg.tp_atr_multiplier, breakeven_trigger_r=None, price_decimals=cfg.price_decimals,
                          risk_pct=cfg.risk_pct, exit_mode="fixed")
            tr = st["_trades"].copy()
            tr["R"] = tr["PnL"] / risk; tr["Side"] = np.where(tr["Size"] > 0, "L", "S")
            tr.insert(0, "Variant", name); tr.insert(0, "Pair", pair)
            rows.append(tr); ons.append(dict(Pair=pair, Variant=name, OnPct=float((al | as_).mean() * 100)))
            if name == CHART_VARIANT:
                chart_trades = st["_trades"]
        fd, ld = day[inp][[0, -1]]
        make_chart(df, out_dir / f"dow_trade_chart_{pair}.html",
                   f"{pair} {fd:%Y-%m-%d} ~ {ld:%Y-%m-%d} / 日足MA20>75>200 + ER(20)≥0.3 + 20MA傾き → 4H {"18本スイング" if ENTRY_MODE == "swing" else "18本高安"}ブレイク SL1.0/TP1.5 ATR",
                   fd, ld, trades=chart_trades, weekly_trend=None, mode="fixed", risk=risk, entry_window=cfg.window, entry_atr=14,
                   price_decimals=cfg.price_decimals, session_start_hour=cfg.session_start_hour, signal_hours=4,
                   filter_state=state_chart, filter_label="日足 MA20/75/200 並び + ER(20)≥0.3 + 20MA傾き（前日確定）",
                   daily_ma=m.MA_PERIODS, daily_er=(20, 0.3), entry_signals=sig, timeframes=("4H", "D1", "W1"))
    (out_dir / f"log_{pair}.txt").write_text(log.getvalue(), encoding="utf-8")
    return pair, pd.concat(rows, ignore_index=True), pd.DataFrame(ons), spread, time.perf_counter() - t0


if __name__ == "__main__":
    out = m.RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{RUN_LABEL}"
    out.mkdir(parents=True, exist_ok=True); (out / "code").mkdir(exist_ok=True)
    here = Path(__file__).resolve().parent
    for fn in (Path(__file__).name, "main_4H_trend_breakout_fast.py", "fast_engine.py"):
        shutil.copy2(here / fn, out / "code" / fn)
    (out / "config.json").write_text(json.dumps(dict(entry_mode=ENTRY_MODE, base_cfg={k: str(v) for k, v in asdict(m.BASE_CFG).items()},
                                                     pairs=m.PAIRS, variants=VARIANTS, warmup_months=m.WARMUP_MONTHS),
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    res = {}
    with ProcessPoolExecutor(m.WORKERS) as ex:
        futs = {ex.submit(run_pair, p, replace(m.BASE_CFG, data_path=m.DATA_ROOT / p, price_decimals=price_decimals_for(p)), out): p
                for p in m.PAIRS}
        for f in as_completed(futs):
            try:
                res[futs[f]] = f.result(); print("ok", futs[f], f"{res[futs[f]][4]:.0f}s", flush=True)
            except Exception as e:
                print("ERR", futs[f], repr(e), flush=True)
    ps = [p for p in m.PAIRS if p in res]
    pd.concat([res[p][1] for p in ps], ignore_index=True).to_csv(out / "trades_all.csv", index=False, encoding="utf-8-sig")
    pd.concat([res[p][2] for p in ps], ignore_index=True).to_csv(out / "filter_on_share.csv", index=False)
    print("saved", out)
