"""
日足ダウ・押し目ブレイク: 日足で高値・安値を切り上げて押し目が確定したら直近高値の上抜けで買い（売りは逆）。

売買ルール（日足は NY 17:00 区切りの取引日。各日の注文は前日の終値で確定した情報だけで決める）
  スイング : dow_trend.py と同じ採用ルール（左右 n 本より厳密に高い/低い日、ATR(14, ワイルダー) × min_swing_atr
             未満の揺れは不採用、高値と安値は交互）。
  買い     : 確定済みスイングの並びが 前の高値 → 前の安値 → 直近の高値 → 直近の安値（押し目）で、
             直近の高値 > 前の高値、直近の安値 > 前の安値、直近の高値・安値とも前日までに抜かれていない、
             かつ 直近の高値 − 直近の安値 が日足 ATR の risk_atr_min〜risk_atr_max 倍
             → 翌日、直近の高値に買い逆指値。損切り = 直近の安値（押し目）。
             その日のうちに押し目の安値に触れたら、その日の注文は取り消す。1日1回まで。
  売り     : 上の逆（高値・安値とも切り下げ、戻り高値が確定した形で、直近の安値に売り逆指値、損切り = 戻り高値）。
  決済     : "trail": 利確なし。新しい押し目の安値（売りは戻り高値）が確定するたびに、翌日から損切りをそこへ引き上げる
             "tp2" / "tp3": 損切りは当初のまま、利確 = エントリー価格 ± 2R / 3R（R = 計画した損切り幅）
  枚数     : 初期資金 × risk_pct ÷ 計画した損切り幅（整数に切り捨て）。ポジションは1ペア1つまで。
約定は1分足で判定し、backtesting.py 0.6 のブローカーと同じ規則にそろえている（compare_dow_daily.py で一致を確認）。
  - 逆指値は High/Low がその価格に触れたら約定、価格は始値と逆指値の不利な方（窓開けは始値）
  - エントリーした1分足では損切りを判定しない。利確は「同じ足で利確に触れ、損切りには触れていない」ときだけ判定する
  - 同じ足で損切りと利確の両方に触れたら損切り
  - 手数料（スプレッド）はエントリーと決済のそれぞれで 枚数 × 価格 × 平均相対スプレッド/2

実行: python dow_daily_breakout.py  → fast_results/<日時>_dow_daily/
グリッド（GRID）の全組み合わせを全ペアで実行し、IS（2021-2023）で1つ選んで OOS（2024-2025）で確かめる。
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dow_trend import DowConfig, atr_wilder, _update_pivots

PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "SP500", "USDCHF", "USDJPY")
START, END = "2021-01-01", "2025-12-31"
IS_END = pd.Timestamp("2023-12-31")
WARMUP_MONTHS = 4                 # 日足スイングの助走期間
CASH = 10_000_000                 # 大きくして枚数の切り捨て誤差を無視できるようにする（成績は R で見る）
RISK_PCT = 0.01
GRID = dict(n=(2, 3), min_swing_atr=(0.5, 1.0), exit=("trail", "tp2", "tp3"))
# 合格ライン: 全期間 PF ≥ 1.2、IS・OOS ともプラス、6ペア中4ペア以上でプラス
PASS_PF, PASS_PAIRS = 1.2, 4
WORKERS = 3
RESULTS_ROOT = Path("fast_results")
RUN_LABEL = "dow_daily"


@dataclass(frozen=True)
class DailyConfig:
    n: int = 2
    min_swing_atr: float = 1.0
    atr_period: int = 14
    risk_atr_min: float = 0.5
    risk_atr_max: float = 3.0
    exit: str = "trail"           # "trail" / "tp2" / "tp3"

    @property
    def tp_r(self):
        return {"trail": None, "tp2": 2.0, "tp3": 3.0}[self.exit]

    @property
    def name(self):
        return f"n{self.n}_sw{self.min_swing_atr}_{self.exit}"


def daily_bars(df_1min, session_start_hour=17):
    """1分足を取引日（NY 17:00 区切り）の日足にまとめる。index = 取引日ラベル"""
    from main_4H_fixedSL import get_trading_day_label
    day = get_trading_day_label(df_1min.index, session_start_hour)
    d1 = df_1min.groupby(day).agg(Open=("Open", "first"), High=("High", "max"),
                                  Low=("Low", "min"), Close=("Close", "last"))
    return d1, day


def daily_setups(d1, cfg: DailyConfig):
    """
    各日の終値時点で決まる「翌日の注文」と、保有中の損切りの引き上げ先。index は d1 と同じ（翌日に使う）。
    列: LongStop, LongSL, ShortStop, ShortSL（その方向の注文が無ければ NaN）、TrailL / TrailH（直近の確定スイング）、
        H1 / L1 / H1Intact / L1Intact（チャート用: 直近のスイングとまだ抜かれていないか）
    """
    o, h, l, c = (d1[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    dcfg = DowConfig(n=cfg.n, atr_period=cfg.atr_period, min_swing_atr=cfg.min_swing_atr)
    atr = atr_wilder(h, l, c, cfg.atr_period)
    out = np.full((len(c), 10), np.nan)
    piv = []
    for i in range(len(c)):
        _update_pivots(piv, i, h, l, atr, dcfg)
        last_h = next((p for p in reversed(piv) if p["kind"] == "H"), None)
        last_l = next((p for p in reversed(piv) if p["kind"] == "L"), None)
        if last_h is None or last_l is None:
            continue
        h_ok = h[last_h["bar"] + 1:i + 1].max(initial=-np.inf) <= last_h["price"]
        l_ok = l[last_l["bar"] + 1:i + 1].min(initial=np.inf) >= last_l["price"]
        out[i, 4], out[i, 5] = last_l["price"], last_h["price"]
        out[i, 6], out[i, 7], out[i, 8], out[i, 9] = last_h["price"], last_l["price"], h_ok, l_ok
        if len(piv) < 4 or np.isnan(atr[i]) or not (h_ok and l_ok):
            continue
        p0, p1, p2, p3 = piv[-4:]
        risk = abs(p2["price"] - p3["price"])
        if not (cfg.risk_atr_min * atr[i] <= risk <= cfg.risk_atr_max * atr[i]):
            continue
        if p3["kind"] == "L" and p2["price"] > p0["price"] and p3["price"] > p1["price"]:
            out[i, 0], out[i, 1] = p2["price"], p3["price"]      # 高値・安値の切り上げ＋押し目
        elif p3["kind"] == "H" and p2["price"] < p0["price"] and p3["price"] < p1["price"]:
            out[i, 2], out[i, 3] = p2["price"], p3["price"]      # 高値・安値の切り下げ＋戻り
    return pd.DataFrame(out, index=d1.index, columns=["LongStop", "LongSL", "ShortStop", "ShortSL", "TrailL",
                                                      "TrailH", "H1", "L1", "H1Intact", "L1Intact"])


def simulate(o, h, l, c, day_start, setups, cfg: DailyConfig, comm, cash=CASH, risk_pct=RISK_PCT):
    """
    day_start: 各取引日の最初の1分足の位置（末尾に全体の本数を足した配列）。setups[k] は k 日目に使う値
    （= 前日の終値で決まったもの。呼び出し側で1日ずらして渡す）。
    戻り値: 取引の dict のリスト（EntryBar, ExitBar, Size, EntryPrice, ExitPrice, SL, TP, PnL, Open=期末に保有中）
    """
    risk_amount = cash * risk_pct
    tp_r = cfg.tp_r
    trades, pos = [], None
    for k in range(len(day_start) - 1):
        a, b = day_start[k], day_start[k + 1]
        s = setups[k]
        if pos is not None and tp_r is None:   # 保有中: 確定した押し目/戻りへ損切りを引き上げる（その日の最初から有効）
            lv = s[4] if pos["Size"] > 0 else s[5]
            if not np.isnan(lv) and (lv > pos["SL"] if pos["Size"] > 0 else lv < pos["SL"]):
                pos["SL"] = lv
        order = None
        if pos is None:
            if not np.isnan(s[0]):
                order = (1, s[0], s[1])
            elif not np.isnan(s[2]):
                order = (-1, s[2], s[3])
        i = a
        while i < b:
            if pos is None:
                if order is None:
                    break
                d, stop, sl = order
                if d > 0:
                    hit = (h[i:b] >= stop) | (l[i:b] <= sl)
                else:
                    hit = (l[i:b] <= stop) | (h[i:b] >= sl)
                if not hit.any():
                    break
                j = i + int(hit.argmax())
                filled = h[j] >= stop if d > 0 else l[j] <= stop
                order = None                       # 約定しても取り消しても、その日はもう注文しない
                if not filled:
                    break
                dist = abs(stop - sl)
                size = int(risk_amount / dist) * d
                if size == 0:
                    break
                price = max(o[j], stop) if d > 0 else min(o[j], stop)
                tp = None if tp_r is None else stop + d * tp_r * dist
                pos = dict(EntryBar=j, Size=size, EntryPrice=price, SL=sl, TP=tp, InitSL=sl,
                           EntryComm=comm(size, price))
                # エントリーした足: 損切りは判定しない。利確は損切りに触れていないときだけ
                if tp is not None and ((d > 0 and h[j] >= tp and l[j] > sl) or (d < 0 and l[j] <= tp and h[j] < sl)):
                    trades.append(_close(pos, j, max(o[j], tp) if d > 0 else min(o[j], tp), comm))
                    pos = None
                i = j + 1
            else:
                d = 1 if pos["Size"] > 0 else -1
                sl, tp = pos["SL"], pos["TP"]
                if d > 0:
                    hit_sl = l[i:b] <= sl
                    hit_tp = h[i:b] >= tp if tp is not None else np.zeros(b - i, bool)
                else:
                    hit_sl = h[i:b] >= sl
                    hit_tp = l[i:b] <= tp if tp is not None else np.zeros(b - i, bool)
                hit = hit_sl | hit_tp
                if not hit.any():
                    break
                j = i + int(hit.argmax())
                if hit_sl[j - i]:
                    price = min(o[j], sl) if d > 0 else max(o[j], sl)
                else:
                    price = max(o[j], tp) if d > 0 else min(o[j], tp)
                trades.append(_close(pos, j, price, comm))
                pos = None
                i = j + 1
    if pos is not None:                            # 期末に保有中: 最後の終値で評価（Open=True）
        t = _close(pos, len(c) - 1, c[-1], comm)
        t["Open"] = True
        trades.append(t)
    return trades


def _close(pos, j, price, comm):
    exit_comm = comm(pos["Size"], price)
    pnl = pos["Size"] * (price - pos["EntryPrice"]) - pos["EntryComm"] - exit_comm
    return dict(EntryBar=pos["EntryBar"], ExitBar=j, Size=pos["Size"], EntryPrice=pos["EntryPrice"],
                ExitPrice=price, SL=pos["SL"], TP=pos["TP"], InitSL=pos["InitSL"], PnL=pnl, Open=False)


def prepare_pair(pair, start=START, end=END):
    """1分足（助走期間込み）、日足、各1分足の取引日、手数料関数を用意する"""
    from main_4H_fixedSL import load_gmo_click_1min_data
    from main_4H_fixedSL_multi import price_decimals_for
    dec = price_decimals_for(pair)
    lo = pd.Timestamp(start).to_period("M") - WARMUP_MONTHS
    hi = pd.Timestamp(end).to_period("M") + 1
    df, spread = load_gmo_click_1min_data(f"histData/{pair}", price_side="mid", price_decimals=dec,
                                          month_range=(lo.year * 100 + lo.month, hi.year * 100 + hi.month),
                                          spread_period=(pd.Timestamp(start), pd.Timestamp(end)))
    d1, day = daily_bars(df)
    rate = spread / 2
    comm = lambda size, price: round(abs(size) * price * rate, dec)  # noqa: E731
    return df, d1, day, comm, dec


def run_config(df, d1, day, comm, cfg, start=START, end=END):
    """1つの設定で検証し、取引履歴の DataFrame と、1分足に割り当てたセットアップ（チャート用）を返す"""
    st = daily_setups(d1, cfg)
    used = st.shift(1)                              # k 日目に使うのは前日の終値で決まった値
    days = d1.index[(d1.index >= pd.Timestamp(start)) & (d1.index <= pd.Timestamp(end))]
    in_period = np.asarray((day >= days[0]) & (day <= days[-1]))
    sub = df.loc[in_period]
    sub_day = day[in_period]
    starts = np.r_[np.flatnonzero(np.r_[True, sub_day[1:] != sub_day[:-1]]), len(sub)]
    day_list = sub_day[starts[:-1]]
    tr = simulate(*(sub[k].to_numpy(float) for k in ("Open", "High", "Low", "Close")), starts,
                  used.reindex(day_list).to_numpy(float), cfg, comm)
    t = pd.DataFrame(tr, columns=["EntryBar", "ExitBar", "Size", "EntryPrice", "ExitPrice", "SL", "TP", "InitSL",
                                  "PnL", "Open"])
    t["EntryTime"] = sub.index[t.EntryBar.to_numpy(int)]
    t["ExitTime"] = sub.index[t.ExitBar.to_numpy(int)]
    t["R"] = t.PnL / (CASH * RISK_PCT)
    return t, used, sub.index, sub_day


def run_pair(pair):
    log = io.StringIO()
    rows = []
    with contextlib.redirect_stdout(log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, d1, day, comm, dec = prepare_pair(pair)
        for n, sw, ex in itertools.product(*GRID.values()):
            cfg = DailyConfig(n=n, min_swing_atr=sw, exit=ex)
            t, *_ = run_config(df, d1, day, comm, cfg)
            t.insert(0, "config", cfg.name)
            t.insert(1, "Pair", pair)
            rows.append(t.drop(columns=["EntryBar", "ExitBar"]))
    (OUT / f"log_{pair}.txt").write_text(log.getvalue(), encoding="utf-8")
    return pair, pd.concat(rows, ignore_index=True)


def make_pair_chart(pair, cfg, out_dir):
    """選んだ設定のトレード付きダウ理論チャート（背景 = その日の注文の向き、ライン = 直近の日足スイング）"""
    from dow_swing_chart import make_chart
    from main_4H_fixedSL import get_trading_day_label
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, d1, day, comm, dec = prepare_pair(pair)
        t, used, idx, sub_day = run_config(df, d1, day, comm, cfg)
        u = used.reindex(day)
        u.index = df.index
        state = pd.Series(np.where(u.LongStop.notna(), 1, np.where(u.ShortStop.notna(), -1, 0)), index=df.index)
        lines = pd.DataFrame({"SignalSH": u.H1, "SignalSL": u.L1,
                              "SignalSHValid": u.H1Intact.fillna(0), "SignalSLValid": u.L1Intact.fillna(0)})
        closed = t[~t.Open]
        first, last = sub_day[0], sub_day[-1]
        path = out_dir / f"dow_trade_chart_{pair}_{cfg.name}.html"
        make_chart(df, path, f"{pair} {first:%Y-%m-%d} ~ {last:%Y-%m-%d} / 日足ダウ・押し目ブレイク {cfg.name}",
                   first, last, trades=closed, weekly_trend=None, mode=cfg.exit, risk=CASH * RISK_PCT,
                   price_decimals=dec, filter_state=state,
                   filter_label=f"日足ダウ（左右{cfg.n}本・ATR×{cfg.min_swing_atr}）: 買いの注文あり / 売りの注文あり",
                   entry_lines=lines)
    return path


def stats(r):
    from main_4H_fixedSL_multi import r_stats
    r = np.asarray(r, float)
    s = r_stats(r)
    cum = np.cumsum(r)
    s["最大DD [R]"] = float((np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max()) if len(r) else 0.0
    return s


def summarize(t):
    """設定ごとの成績（期末に保有中のトレードは含めない）と、合格ラインの判定"""
    t = t[~t.Open].sort_values(["ExitTime", "EntryTime"], kind="stable")
    rows = []
    for name, g in t.groupby("config", sort=False):
        s = dict(config=name, **stats(g.R))
        for part, m in (("IS", g.IS), ("OOS", ~g.IS)):
            ps = stats(g.loc[m, "R"])
            s.update({f"{part}_n": ps["トレード数"], f"{part}_R": ps["合計R"], f"{part}_PF": ps["プロフィットファクター"]})
        pr = g.groupby("Pair").R.sum()
        s["plus_pairs"] = int((pr > 0).sum())
        s["long_R"], s["short_R"] = g.loc[g.Size > 0, "R"].sum(), g.loc[g.Size < 0, "R"].sum()
        s["pass"] = bool(s["プロフィットファクター"] >= PASS_PF and s["IS_R"] > 0 and s["OOS_R"] > 0
                         and s["plus_pairs"] >= PASS_PAIRS)
        rows.append(s)
    return pd.DataFrame(rows)


OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{RUN_LABEL}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for pair, t in ex.map(run_pair, PAIRS):
            print(pair, len(t), flush=True)
            parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    for c in ("EntryTime", "ExitTime"):
        t[c] = pd.to_datetime(t[c], utc=True).dt.tz_convert("America/New_York")
    t["IS"] = t.EntryTime.dt.tz_localize(None) <= IS_END + pd.Timedelta(days=1)
    t["Year"] = t.ExitTime.dt.year
    t.round({"R": 4}).to_csv(OUT / "trades_all.csv", index=False, encoding="utf-8-sig")

    summ = summarize(t)
    # IS（2021-2023）の合計Rが最大の設定を1つ選ぶ。OOS はこの設定で確かめる
    pick = summ.sort_values("IS_R", ascending=False).iloc[0]["config"]
    summ["picked"] = summ.config == pick
    summ.round(4).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    closed = t[~t.Open]
    yearly = closed.pivot_table(index="config", columns="Year", values="R", aggfunc="sum").round(3)
    by_pair = closed.pivot_table(index="config", columns="Pair", values="R", aggfunc="sum").round(3)
    yearly.to_csv(OUT / "yearly.csv", encoding="utf-8-sig")
    by_pair.to_csv(OUT / "by_pair.csv", encoding="utf-8-sig")

    cfg = next(DailyConfig(n=n, min_swing_atr=sw, exit=ex) for n, sw, ex in itertools.product(*GRID.values())
               if DailyConfig(n=n, min_swing_atr=sw, exit=ex).name == pick)
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(make_pair_chart, PAIRS, [cfg] * len(PAIRS), [OUT] * len(PAIRS)))

    info = dict(run_label=RUN_LABEL, started_at=datetime.now().isoformat(timespec="seconds"), pairs=PAIRS,
                period=[START, END], is_end=str(IS_END.date()), cash=CASH, risk_pct=RISK_PCT, grid=GRID,
                defaults=dataclasses.asdict(DailyConfig()), pass_rule=dict(pf=PASS_PF, plus_pairs=PASS_PAIRS),
                picked=pick)
    (OUT / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    code = OUT / "code"
    code.mkdir(exist_ok=True)
    for f in (Path(__file__).name, "dow_trend.py", "main_4H_fixedSL.py"):
        shutil.copy2(Path(__file__).with_name(f), code / f)

    pd.set_option("display.width", 250)
    cols = ["config", "トレード数", "勝率 [%]", "合計R", "プロフィットファクター", "最大DD [R]", "IS_n", "IS_R", "OOS_n",
            "OOS_R", "plus_pairs", "long_R", "short_R", "pass", "picked"]
    print(summ[cols].round(2).to_string(index=False))
    print(yearly.round(1).to_string())
    print(by_pair.round(1).to_string())
    print(f"期末に保有中で集計から除いたトレード: {int(t.Open.sum())}")
    print(f"\n結果: {OUT}")


if __name__ == "__main__":
    main()
