"""
日足ダウの方向 × 4時間足ブレイク: 日足で高値・安値を切り上げている日は4時間足の高値ブレイクで買いだけ、
切り下げている日は安値ブレイクで売りだけ。どちらでもない日は見送る。

売買ルール
  日足の方向（前日の終値で確定した情報。dow_trend.swing_structure と同じスイング）
    買い許可: 直近のスイングハイ > 1つ前のスイングハイ かつ 直近のスイングロー > 1つ前のスイングロー、
              かつ 直近のスイングローがまだ割られていない
    売り許可: 上の逆（直近のスイングハイがまだ超えられていない）
  4時間足のエントリーライン（1本前の4時間足の終値で確定した情報）
    "h4dow": dow_trend と同じ採用ルールの直近スイング（左右3本、ATR(14) × 1.0）。まだ抜かれていないものだけ
    "h4w18": 従来のスイング（前後18本の最高値・最安値。main_4H_fixedSL.compute_signals）
    買い許可の日はスイングハイに買い逆指値、売り許可の日はスイングローに売り逆指値（fast_engine の AllowLong/AllowShort）
  決済・枚数: fast_engine の2通り（backtesting.py との一致は確認済み）
    "fixed": SL 1.5 / TP 2.5 × ATR(EMA 18)、建値ストップなし
    "trail": 1時間足スイング（前後5本）へのトレーリング、当初SL 1.5 × ATR、TP なし
    1回の損失 = 初期資金 × 2%（1R）。スプレッドは平均スプレッドを手数料で近似。

GRID の全組み合わせと、日足の方向を使わない比較用（d1="none"）を全ペアで実行する。
IS（2021-2023）の合計R が最大の設定を1つ選び、OOS（2024-2025）で確かめる。
結果は fast_results/<日時>_d1dow_h4/ に保存する。
"""
import contextlib
import dataclasses
import io
import itertools
import json
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_fixedSL import (resample_to_signal_bars, compute_signals, map_signals_to_1min, get_trading_day_label)
from main_4H_fixedSL_multi import compute_h1_swings
from main_4H_fixedSL_fast_multi import BASE_CFG
from fast_engine import run_fast
from dow_trend import DowConfig, swing_structure
from dow_structure_breakout import structure_signals
from dow_daily_breakout import prepare_pair, daily_bars, stats, PASS_PF, PASS_PAIRS

PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "SP500", "USDCHF", "USDJPY")
START, END = BASE_CFG.start_date, BASE_CFG.end_date
IS_END = pd.Timestamp("2023-12-31")
GRID = dict(d1=("n2", "n3"), entry=("h4dow", "h4w18"), exit=("fixed", "trail"))
D1_CFG = {"n2": DowConfig(n=2, atr_period=14, min_swing_atr=1.0), "n3": DowConfig(n=3, atr_period=14, min_swing_atr=1.0)}
H4_DOW = DowConfig(n=3, atr_period=14, min_swing_atr=1.0)
MODES = {"fixed": dict(exit_mode="fixed"), "trail": dict(exit_mode="swing_trail", initial_sl_rule="atr")}
MODE_TEXT = {"fixed": "固定SL 1.5 / TP 2.5 ATR", "trail": "1Hスイング追従（前後5本・当初SL 1.5 ATR）"}
ENTRY_TEXT = {"h4dow": "4Hダウのスイング（左右3本）", "h4w18": "4Hスイング（前後18本）"}
H1_WINDOW = 5
WORKERS = 3
RUN_LABEL = "d1dow_h4"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fast_results") / f"{datetime.now():%Y%m%d_%H%M}_{RUN_LABEL}"


def cfg_name(d1, entry, ex):
    return f"{d1}_{entry}_{ex}"


def daily_direction(df_1min, d1, day, dcfg):
    """1分足ごとの日足の方向（1=買いだけ / -1=売りだけ / 0=見送り）。前日の終値で確定した値"""
    st = swing_structure(d1, dcfg)
    up = (st.H1 > st.H0) & (st.L1 > st.L0) & (st.L1Intact == 1)
    dn = (st.H1 < st.H0) & (st.L1 < st.L0) & (st.H1Intact == 1)
    state = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=d1.index).shift(1).fillna(0)
    return pd.Series(state.reindex(day).to_numpy(), index=df_1min.index)


def run_pair(pair, make_charts_for=None):
    """make_charts_for: (d1, entry, exit) を渡すと、その設定のチャートも出力する"""
    log = io.StringIO()
    rows = []
    with contextlib.redirect_stdout(log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_1min, d1, day, comm, dec = prepare_pair(pair, START, END)
        cfg = dataclasses.replace(BASE_CFG, price_decimals=dec)
        h4 = resample_to_signal_bars(df_1min, 17, 4)
        sig = compute_signals(h4, window=cfg.window, atr_period=cfg.atr_period, price_decimals=dec)
        base = map_signals_to_1min(df_1min, sig, 17, 4).join(compute_h1_swings(df_1min, H1_WINDOW, 17))
        in_period = np.asarray((day >= pd.Timestamp(START)) & (day <= pd.Timestamp(END)))
        first_day, last_day = day[in_period][[0, -1]]
        h4dow = structure_signals(df_1min, h4, H4_DOW)
        lines = {"h4w18": None, "h4dow": h4dow}
        dirs = {k: daily_direction(df_1min, d1, day, c) for k, c in D1_CFG.items()}
        risk = cfg.cash * cfg.risk_pct

        combos = [("none", e, x) for e in GRID["entry"] for x in GRID["exit"]] + list(itertools.product(*GRID.values()))
        if make_charts_for:
            combos = [make_charts_for]
        for d1k, entry, ex in combos:
            d = base.copy()
            if entry == "h4dow":
                for c in ("SignalSH", "SignalSL", "SignalSHValid", "SignalSLValid"):
                    d[c] = h4dow[c].to_numpy(float)
            if d1k != "none":
                s = dirs[d1k].to_numpy()
                d["AllowLong"], d["AllowShort"] = s == 1, s == -1
            d = d.loc[in_period]
            params = dict(sl_atr_multiplier=cfg.sl_atr_multiplier, price_decimals=dec, risk_pct=cfg.risk_pct,
                          **MODES[ex])
            if ex == "fixed":
                params.update(tp_atr_multiplier=cfg.tp_atr_multiplier, breakeven_trigger_r=None)
            st = run_fast(d, cash=cfg.cash, commission=comm, margin=cfg.margin, **params)
            tr = st["_trades"]
            for _, x in tr.iterrows():
                rows.append(dict(config=cfg_name(d1k, entry, ex), d1=d1k, entry=entry, exit=ex, Pair=pair,
                                 Side="L" if x.Size > 0 else "S", EntryTime=x.EntryTime, ExitTime=x.ExitTime,
                                 EntryPrice=x.EntryPrice, ExitPrice=x.ExitPrice, SL=x.SL, TP=x.TP, R=x.PnL / risk))
            if make_charts_for:
                from dow_swing_chart import make_chart
                state = dirs[d1k]
                make_chart(df_1min, OUT / f"dow_trade_chart_{pair}_{cfg_name(d1k, entry, ex)}.html",
                           f"{pair} {first_day:%Y-%m-%d} ~ {last_day:%Y-%m-%d} / 日足ダウ({d1k}) × {ENTRY_TEXT[entry]} / "
                           f"{MODE_TEXT[ex]}",
                           first_day, last_day, trades=tr, weekly_trend=None, mode=MODES[ex]["exit_mode"], risk=risk,
                           entry_window=cfg.window, entry_atr=cfg.atr_period, price_decimals=dec,
                           filter_state=state,
                           filter_label=f"日足ダウ（左右{D1_CFG[d1k].n}本）: 切り上げ=買いだけ / 切り下げ=売りだけ",
                           entry_lines=lines[entry])
    if not make_charts_for:
        (OUT / f"log_{pair}.txt").write_text(log.getvalue(), encoding="utf-8")
    return pair, rows


def chart_job(args):
    return run_pair(*args)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for pair, r in ex.map(run_pair, PAIRS):
            print(pair, len(r), flush=True)
            rows += r
    t = pd.DataFrame(rows)
    for c in ("EntryTime", "ExitTime"):
        t[c] = pd.to_datetime(t[c], utc=True).dt.tz_convert("America/New_York")
    t = t.sort_values(["ExitTime", "EntryTime"], kind="stable").reset_index(drop=True)
    t["IS"] = t.EntryTime.dt.tz_localize(None) <= IS_END + pd.Timedelta(days=1)
    t["Year"] = t.ExitTime.dt.year
    t.round({"R": 4}).to_csv(OUT / "trades_all.csv", index=False, encoding="utf-8-sig")

    summ = []
    for name, g in t.groupby("config", sort=False):
        s = dict(config=name, d1=g.d1.iloc[0], entry=g.entry.iloc[0], exit=g.exit.iloc[0], **stats(g.R))
        for part, m in (("IS", g.IS), ("OOS", ~g.IS)):
            ps = stats(g.loc[m, "R"])
            s.update({f"{part}_n": ps["トレード数"], f"{part}_R": ps["合計R"], f"{part}_PF": ps["プロフィットファクター"]})
        s["plus_pairs"] = int((g.groupby("Pair").R.sum() > 0).sum())
        s["long_R"], s["short_R"] = g.loc[g.Side == "L", "R"].sum(), g.loc[g.Side == "S", "R"].sum()
        s["exJPY_R"] = g.loc[g.Pair != "USDJPY", "R"].sum()
        s["pass"] = bool(s["プロフィットファクター"] >= PASS_PF and s["IS_R"] > 0 and s["OOS_R"] > 0
                         and s["plus_pairs"] >= PASS_PAIRS)
        summ.append(s)
    summ = pd.DataFrame(summ).sort_values(["d1", "entry", "exit"]).reset_index(drop=True)
    cand = summ[summ.d1 != "none"]
    pick = cand.sort_values("IS_R", ascending=False).iloc[0]
    summ["picked"] = summ.config == pick.config
    summ.round(4).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    t.pivot_table(index="config", columns="Year", values="R", aggfunc="sum").round(3).to_csv(
        OUT / "yearly.csv", encoding="utf-8-sig")
    t.pivot_table(index="config", columns="Pair", values="R", aggfunc="sum").round(3).to_csv(
        OUT / "by_pair.csv", encoding="utf-8-sig")

    target = (pick.d1, pick.entry, pick.exit)
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(chart_job, [(p, target) for p in PAIRS]))

    info = dict(run_label=RUN_LABEL, started_at=datetime.now().isoformat(timespec="seconds"), pairs=PAIRS,
                period=[START, END], is_end=str(IS_END.date()), grid=GRID,
                d1_cfg={k: dataclasses.asdict(v) for k, v in D1_CFG.items()}, h4_dow=dataclasses.asdict(H4_DOW),
                modes=MODES, h1_window=H1_WINDOW, pass_rule=dict(pf=PASS_PF, plus_pairs=PASS_PAIRS),
                picked=pick.config,
                base_cfg={k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(BASE_CFG).items()})
    (OUT / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    code = OUT / "code"
    code.mkdir(exist_ok=True)
    for f in (Path(__file__).name, "dow_daily_breakout.py", "dow_structure_breakout.py", "dow_trend.py",
              "fast_engine.py", "main_4H_fixedSL.py"):
        shutil.copy2(Path(__file__).with_name(f), code / f)

    pd.set_option("display.width", 260)
    cols = ["config", "トレード数", "勝率 [%]", "合計R", "平均R", "プロフィットファクター", "最大DD [R]", "IS_R", "OOS_R",
            "plus_pairs", "long_R", "short_R", "exJPY_R", "pass", "picked"]
    print(summ[cols].round(2).to_string(index=False))
    print(t.pivot_table(index="config", columns="Year", values="R", aggfunc="sum").round(1).to_string())
    print(t.pivot_table(index="config", columns="Pair", values="R", aggfunc="sum").round(1).to_string())
    print(f"\n結果: {OUT}")


if __name__ == "__main__":
    main()
