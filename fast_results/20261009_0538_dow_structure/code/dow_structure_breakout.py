"""
ダウ理論のスイング構造ブレイクアウト（高値・安値を切り上げていたら高値超えで買い、切り下げていたら安値割れで売り）を
fast_engine で全ペア検証し、従来のスイングブレイクアウト（4時間足 前後18本、フィルターなし）と比べる。

売買ルール（4時間足。各足の値は1本前の4時間足の終値時点で確定していた情報だけを使う）
  - スイングは dow_trend.py と同じ採用ルール（左右 n 本より厳密に高い/低い足、最小スイング幅 ATR(14)×min_swing_atr、
    H と L が交互）。H1 / H0 = 直近 / 1つ前のスイングハイ、L1 / L0 = 直近 / 1つ前のスイングロー。
  - 買い: H1 > H0 かつ L1 > L0（高値・安値とも切り上げ）、かつ L1 がまだ割られていない
          → H1 がまだ超えられていなければ H1 に買い逆指値
  - 売り: H1 < H0 かつ L1 < L0（高値・安値とも切り下げ）、かつ H1 がまだ超えられていない
          → L1 がまだ割られていなければ L1 に売り逆指値
  - 決済・枚数は従来と同じ2通り（fast_engine の exit_mode）:
      fixed: SL 1.5 / TP 2.5 × ATR(EMA 18)、建値ストップなし
      trail: 1時間足スイング（前後5本）へのトレーリング、当初SL 1.5 × ATR、TP なし
  - 1回の損失 = 初期資金 × 2%（1R）。スプレッドは従来と同じく平均スプレッドを手数料で近似。

結果は fast_results/<日時>_<RUN_LABEL>/ に保存する（trades_all.csv、summary.csv、yearly.csv、by_pair.csv、
equity_R.csv、config.json、code/、dow_trade_chart_<ペア>_<モード>.html）。
IS = 2021-2023、OOS = 2024-2025 に分けても集計する（全期間で1回実行し、エントリー日で分ける）。
"""
import contextlib
import dataclasses
import io
import json
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_fixedSL import (load_gmo_click_1min_data, resample_to_signal_bars, compute_signals,
                             map_signals_to_1min, get_trading_day_label, get_signal_bar_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for, compute_h1_swings, r_stats
from main_4H_fixedSL_fast_multi import BASE_CFG
from fast_engine import run_fast
from dow_trend import DowConfig, swing_structure
from dow_swing_chart import make_chart

# ===== 変更するパラメータはここだけ =====
PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "SP500", "USDCHF", "USDJPY")
# 検証するスイングの設定（名前: DowConfig）。"base" は従来のスイングブレイクアウト（比較用、常に実行）
VARIANTS = {
    "dow_n3": DowConfig(n=3, atr_period=14, min_swing_atr=1.0),
    "dow_n5": DowConfig(n=5, atr_period=14, min_swing_atr=1.0),
}
MODES = {"fixed": dict(exit_mode="fixed"), "trail": dict(exit_mode="swing_trail", initial_sl_rule="atr")}
MODE_TEXT = {"fixed": "固定SL 1.5 / TP 2.5 ATR", "trail": "1Hスイング追従（前後5本・当初SL 1.5 ATR）"}
H1_WINDOW = 5
IS_END = pd.Timestamp("2023-12-31")
CHART_VARIANT = "dow_n3"          # トレード付きダウ理論チャートを出す設定（ペア × 決済ルールごと）
RUN_LABEL = "dow_structure"
WORKERS = 3
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fast_results") / f"{datetime.now():%Y%m%d_%H%M}_{RUN_LABEL}"


def structure_signals(df_1min, sig_4h, dcfg):
    """4時間足のスイング構造から、1分足ごとのエントリーライン・有効フラグ・方向の許可を作る（1本ずらし済み）"""
    st = swing_structure(sig_4h, dcfg).shift(1)
    up = (st.H1 > st.H0) & (st.L1 > st.L0) & (st.L1Intact == 1)
    dn = (st.H1 < st.H0) & (st.L1 < st.L0) & (st.H1Intact == 1)
    m = pd.DataFrame({"SignalSH": st.H1, "SignalSL": st.L1,
                      "SignalSHValid": st.H1Intact.fillna(0), "SignalSLValid": st.L1Intact.fillna(0),
                      "AllowLong": up, "AllowShort": dn}, index=st.index)
    mapped = m.reindex(get_signal_bar_label(df_1min.index, 17, 4))
    mapped.index = df_1min.index
    return mapped


def run_pair(pair):
    log = io.StringIO()
    rows, curves = [], {}
    with contextlib.redirect_stdout(log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = dataclasses.replace(BASE_CFG, data_path=f"histData/{pair}", price_decimals=price_decimals_for(pair))
        start_ts, end_ts = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df_1min, spread = load_gmo_click_1min_data(cfg.data_path, price_side=cfg.price_side,
                                                   price_decimals=cfg.price_decimals,
                                                   month_range=zip_month_range(cfg),
                                                   spread_period=(start_ts, end_ts))
        h4 = resample_to_signal_bars(df_1min, 17, 4)
        sig = compute_signals(h4, window=cfg.window, atr_period=cfg.atr_period, price_decimals=cfg.price_decimals)
        base = map_signals_to_1min(df_1min, sig, 17, 4).join(compute_h1_swings(df_1min, H1_WINDOW, 17))
        day = get_trading_day_label(base.index, 17)
        in_period = np.asarray((day >= start_ts) & (day <= end_ts))
        first_day, last_day = day[in_period][[0, -1]]
        rate = spread / 2
        comm = lambda size, price: round(abs(size) * price * rate, cfg.price_decimals)  # noqa: E731
        risk = cfg.cash * cfg.risk_pct

        frames = {"base": (base.loc[in_period], None)}
        for name, dcfg in VARIANTS.items():
            s = structure_signals(df_1min, h4, dcfg)
            d = base.copy()
            for c in ("SignalSH", "SignalSL", "SignalSHValid", "SignalSLValid"):
                d[c] = s[c].to_numpy(float)
            d["AllowLong"] = s.AllowLong.to_numpy(bool)
            d["AllowShort"] = s.AllowShort.to_numpy(bool)
            frames[name] = (d.loc[in_period], s)

        for name, (d, s) in frames.items():
            for mode, mp in MODES.items():
                params = dict(sl_atr_multiplier=cfg.sl_atr_multiplier, price_decimals=cfg.price_decimals,
                              risk_pct=cfg.risk_pct, **mp)
                if mode == "fixed":
                    params.update(tp_atr_multiplier=cfg.tp_atr_multiplier, breakeven_trigger_r=None)
                st = run_fast(d, cash=cfg.cash, commission=comm, margin=cfg.margin, **params)
                tr = st["_trades"]
                for _, x in tr.iterrows():
                    rows.append(dict(variant=name, mode=mode, Pair=pair, Side="L" if x.Size > 0 else "S",
                                     EntryTime=x.EntryTime, ExitTime=x.ExitTime, EntryPrice=x.EntryPrice,
                                     ExitPrice=x.ExitPrice, SL=x.SL, TP=x.TP, R=x.PnL / risk))
                eq = (st["_equity_curve"]["Equity"] - cfg.cash) / risk
                curves[(name, mode)] = eq.resample("1D").last().dropna()
                if name == CHART_VARIANT:
                    dcfg = VARIANTS[name]
                    state = pd.Series(np.where(s.AllowLong, 1, np.where(s.AllowShort, -1, 0)), index=s.index)
                    make_chart(df_1min, OUT / f"dow_trade_chart_{pair}_{mode}.html",
                               f"{pair} {first_day:%Y-%m-%d} ~ {last_day:%Y-%m-%d} / ダウ構造ブレイク(4H n={dcfg.n}) / "
                               f"{MODE_TEXT[mode]}",
                               first_day, last_day, trades=tr, weekly_trend=None, mode=mp["exit_mode"], risk=risk,
                               price_decimals=cfg.price_decimals, session_start_hour=17, signal_hours=4,
                               filter_state=state,
                               filter_label=f"4Hダウ構造(n={dcfg.n}): 高値・安値の切り上げ=買い / 切り下げ=売り",
                               entry_lines=s)
    (OUT / f"log_{pair}.txt").write_text(log.getvalue(), encoding="utf-8")
    return pair, rows, curves


def max_dd(cum):
    cum = np.asarray(cum, float)
    return float((np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max()) if len(cum) else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    with ProcessPoolExecutor(WORKERS) as ex:
        for pair, r, c in ex.map(run_pair, PAIRS):
            print(pair, len(r), flush=True)
            rows += r
            curves.update({(*k, pair): v for k, v in c.items()})
    t = pd.DataFrame(rows)
    for c in ("EntryTime", "ExitTime"):
        t[c] = pd.to_datetime(t[c], utc=True).dt.tz_convert("America/New_York")
    t = t.sort_values(["ExitTime", "EntryTime"], kind="stable").reset_index(drop=True)
    t["IS"] = t.EntryTime.dt.tz_localize(None) <= IS_END + pd.Timedelta(days=1)
    t["Year"] = t.ExitTime.dt.year
    t.round({"R": 4}).to_csv(OUT / "trades_all.csv", index=False, encoding="utf-8-sig")

    summ, yearly, by_pair, eq_all = [], [], [], {}
    for (v, mode), g in t.groupby(["variant", "mode"], sort=False):
        # 合算の資産曲線（含み損益込み・日次）から最大DDを出す
        eq = (pd.concat({p: curves[(v, mode, p)] for p in PAIRS if (v, mode, p) in curves}, axis=1, sort=True)
              .sort_index().ffill().fillna(0).sum(axis=1))
        eq_all[f"{v}|{mode}"] = eq
        s = dict(variant=v, mode=mode, **r_stats(g.R), maxDD_R=float((eq.cummax() - eq).max()),
                 maxDD_closed_R=max_dd(g.R.cumsum()))
        for part, m in (("IS", g.IS), ("OOS", ~g.IS)):
            rs = r_stats(g.loc[m, "R"])
            s.update({f"{part}_n": rs["トレード数"], f"{part}_R": rs["合計R"], f"{part}_PF": rs["プロフィットファクター"]})
        s["long_R"] = g.loc[g.Side == "L", "R"].sum()
        s["short_R"] = g.loc[g.Side == "S", "R"].sum()
        summ.append(s)
        for y, gy in g.groupby("Year"):
            yearly.append(dict(variant=v, mode=mode, year=int(y), R=gy.R.sum(), n=len(gy)))
        for p, gp in g.groupby("Pair"):
            rs = r_stats(gp.R)
            by_pair.append(dict(variant=v, mode=mode, pair=p, n=rs["トレード数"], R=rs["合計R"],
                                win=rs["勝率 [%]"], PF=rs["プロフィットファクター"]))
    summ, yearly, by_pair = pd.DataFrame(summ), pd.DataFrame(yearly), pd.DataFrame(by_pair)
    summ.round(4).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    yearly.round(4).to_csv(OUT / "yearly.csv", index=False, encoding="utf-8-sig")
    by_pair.round(4).to_csv(OUT / "by_pair.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eq_all).round(3).to_csv(OUT / "equity_R.csv", encoding="utf-8-sig")

    info = dict(run_label=RUN_LABEL, started_at=datetime.now().isoformat(timespec="seconds"), pairs=PAIRS,
                variants={k: dataclasses.asdict(v) for k, v in VARIANTS.items()}, modes=MODES, h1_window=H1_WINDOW,
                is_end=str(IS_END.date()),
                base_cfg={k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(BASE_CFG).items()})
    (OUT / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    code = OUT / "code"
    code.mkdir(exist_ok=True)
    for f in (Path(__file__).name, "fast_engine.py", "dow_trend.py", "main_4H_fixedSL.py"):
        shutil.copy2(Path(__file__).with_name(f), code / f)

    pd.set_option("display.width", 250)
    print(summ.round(2).to_string(index=False))
    print(yearly.pivot_table(index=["variant", "mode"], columns="year", values="R").round(1).to_string())
    print(by_pair.pivot_table(index=["variant", "mode"], columns="pair", values="R").round(1).to_string())
    print(f"\n結果: {OUT}")


if __name__ == "__main__":
    main()
