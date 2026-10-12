"""
上位足押し目 × 下位足転換 戦略（ロングのみ）の検証。仕様は docs/MTF_PULLBACK_SPEC.md。

  上位足が上昇トレンド（スイング高値・安値の切り上げ）で、上位足のサポートゾーン（抜けた前回高値）まで押した。
  そこで下位足の下降構造（高値の切り下げ）が崩れ（CHoCH）、下位足のロールリバーサル水準（H_b）まで押し、
  トリガー（直近 trigger_n 本の高値を終値で上抜け）を確認して、次の足の始値で買う。

- 入力は1分足（GMOクリック証券）。下位足（ltf_hours 時間足）と上位足（htf_rule）は NY 17:00 区切りで作る
  （上位足は下位足をまとめて作るので、両者の区切りは必ず揃う）
- シグナルは下位足の確定（終値）で判定する。上位足の状態は、その上位足が確定した後の下位足にだけ使う。
  スイングは前後 n 本のフラクタルで、ピボットの n 本後に初めて使う（未来参照の防止）
- 約定（エントリー後の SL/TP）は既定で1分足で判定する（exec_1min=True）。同じ足で SL と TP の両方に触れたら SL を優先。
  窓開けで SL / TP を飛び越えたときは、その足の始値で約定する
- スプレッドは既存スクリプトと同じく、平均相対スプレッドの半分をエントリーと決済のそれぞれに課金する（R は課金後）

実行（リポジトリのルートで）:  python mtf_pullback_strategy.py
結果は mtf_results/<日時>_<ラベル>/ に保存する（前回の結果を上書きしない）。
"""
import bisect
import dataclasses
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_fixedSL import (
    calculate_atr_ema, get_signal_bar_label, get_trading_day_label, load_gmo_click_1min_data,
    resample_to_signal_bars,
)


# ===== 変更するパラメータはここだけ =====
@dataclass(frozen=True)
class MtfConfig:
    pair: str = "EURUSD"
    data_root: Path = Path("histData")
    start_date: str | None = "2021-01-01"   # 取引日（NY 17:00 区切り）で絞り込む。前の WARMUP_MONTHS ヶ月は助走期間
    end_date: str | None = "2025-12-31"
    session_start_hour: int = 17
    ltf_hours: int = 1                # 下位足（時間）。24 の約数
    htf_rule: str = "4h"              # 上位足: "4h" / "1D" / "1W"（など "<N>h"。24 の約数）
    swing_n_htf: int = 3              # 上位足スイング判定の左右本数
    swing_n_ltf: int = 3              # 下位足スイング判定の左右本数
    atr_len: int = 14                 # ATR の期間（上位足・下位足とも。EMA の ATR）
    support_mode: str = "broken_high"  # "broken_high": 上抜けされた前回スイング高値 / "last_low": 直近の押し安値
    htf_zone_atr: float = 0.5         # 上位足ゾーン = 水準 ± 上位足ATR × これ
    rr_away_atr: float = 1.0          # 転換後に RR 水準から上へ離れる幅（下位足ATR倍）
    rr_zone_atr: float = 0.3          # RR 水準への接触判定幅（下位足ATR倍）
    trigger_n: int = 5                # トリガー: 終値が直近 trigger_n 本の高値を上抜け
    setup_expiry: int = 72            # 各段階の有効期限（下位足本数）
    sl_buffer_atr: float = 0.2        # 損切り = 接触安値 − ATR × これ
    rr_target: float | None = 2.0     # 利確 = エントリー + 損切り幅 × これ（None なら上位足の目標値）
    sides: tuple = ("long", "short")   # ショートは価格の高値と安値を反転（符号を反転）したロジックを同じコードで走らせる
    exec_1min: bool = True            # True: SL/TP を1分足で判定 / False: 下位足で判定
    price_decimals: int = 5
    cash: float = 10_000              # チャートの損益表示用（1R = cash × risk_pct）
    risk_pct: float = 0.02


CFG = MtfConfig(htf_rule="1D")
WARMUP_MONTHS = 3
RESULTS_ROOT = Path("mtf_results")
RUN_LABEL = None                      # None なら「ペア_上位足x下位足」


# ---------------------------------------------------------------------------
# 足の作成
# ---------------------------------------------------------------------------
def htf_labels(ltf_index, htf_rule, session_start_hour=17):
    """下位足の開始時刻ラベル（NY 現地時刻・TZ なし）→ その足が属する上位足の開始時刻ラベル"""
    rule = htf_rule.lower()
    off = pd.Timedelta(hours=session_start_hour)
    idx = pd.DatetimeIndex(ltf_index)
    if rule.endswith("w"):
        # 週足: 日曜 17:00 起点。取引日（翌日側）の週の月曜で丸めてから起点の時刻に戻す
        day = (idx - off).normalize() + pd.Timedelta(days=1)
        monday = day - pd.to_timedelta(day.weekday, unit="D")
        return monday - pd.Timedelta(days=1) + off
    hours = 24 if rule.endswith("d") else int(rule.rstrip("h"))
    if 24 % hours:
        raise ValueError(f"htf_rule は 24 の約数の時間足か 1D / 1W: {htf_rule}")
    return (idx - off).floor(pd.Timedelta(hours=hours)) + off


PRICE_COLS = ("entry", "sl", "tp", "exit", "rr_level", "swing_low", "h_a", "bottom", "inval", "htf_target",
              "contact_price")


def mirror(df):
    """価格を符号反転し、高値と安値を入れ替える（ショートをロングのロジックで扱うため。足の位置・ATR は変わらない）"""
    out = df.copy()
    out["Open"], out["Close"] = -df["Open"], -df["Close"]
    out["High"], out["Low"] = -df["Low"], -df["High"]
    return out


def unmirror_trades(trades):
    """mirror() した価格で作ったトレード一覧を、実際の価格（ショート）に戻す。R・位置の列はそのまま"""
    t = trades.copy()
    for c in PRICE_COLS:
        t[c] = -t[c]
    t["zone_lo"], t["zone_hi"] = -trades["zone_hi"], -trades["zone_lo"]
    t["side"] = "short"
    return t


def resample_bars(df, labels):
    return df.groupby(labels).agg(Open=("Open", "first"), High=("High", "max"),
                                  Low=("Low", "min"), Close=("Close", "last"))


# ---------------------------------------------------------------------------
# スイング（フラクタル）
# ---------------------------------------------------------------------------
def fractal_pivots(high, low, n):
    """
    左右 n 本のフラクタル。同値が並んだときは最初の1本だけ:
      高値: 左 n 本の最大 < 当該高値 かつ 当該高値 >= 右 n 本の最大（安値は逆）
    返り値: (スイング高値のピボット位置の配列, スイング安値のピボット位置の配列)。確定はピボットの n 本後
    """
    high, low = np.asarray(high, float), np.asarray(low, float)
    m = len(high)
    if m < 2 * n + 1:
        return np.array([], int), np.array([], int)
    w = np.lib.stride_tricks.sliding_window_view
    hw, lw = w(high, n), w(low, n)               # hw[i] = high[i:i+n]
    j = np.arange(n, m - n)
    left_h, right_h = hw[j - n].max(1), hw[j + 1].max(1)
    left_l, right_l = lw[j - n].min(1), lw[j + 1].min(1)
    sh = j[(left_h < high[j]) & (high[j] >= right_h)]
    sl = j[(left_l > low[j]) & (low[j] <= right_l)]
    return sh, sl


# ---------------------------------------------------------------------------
# 上位足の状態
# ---------------------------------------------------------------------------
def htf_context(htf, cfg):
    """
    上位足の各足の確定時点の状態（その足までに確定したスイングだけを使う）:
      Up（上昇トレンド）, ZoneLo / ZoneHi（サポートゾーン）, Level（ゾーンの中心）, Target（直近スイング高値）, ATR
    """
    h, l, c = (htf[k].to_numpy(float) for k in ("High", "Low", "Close"))
    n = cfg.swing_n_htf
    atr = calculate_atr_ema(h, l, c, cfg.atr_len, cfg.price_decimals)
    sh_piv, sl_piv = fractal_pivots(h, l, n)
    sh_at = {p + n: p for p in sh_piv}            # 確定した足 → ピボット
    sl_at = {p + n: p for p in sl_piv}
    m = len(htf)
    up = np.zeros(m, bool)
    level = np.full(m, np.nan)
    target = np.full(m, np.nan)
    highs, lows = [], []                          # 確定済みスイングの価格
    unbroken = []                                 # まだ終値で上抜けされていないスイング高値 [(ピボット, 価格)]
    broken = np.nan                               # 直近（ピボットが新しい側）に上抜けされたスイング高値
    broken_piv = -1
    for i in range(m):
        if i in sh_at:
            highs.append(h[sh_at[i]])
            unbroken.append((sh_at[i], h[sh_at[i]]))
        if i in sl_at:
            lows.append(l[sl_at[i]])
        keep = []
        for piv, price in unbroken:
            if c[i] > price:
                if piv > broken_piv:
                    broken, broken_piv = price, piv
            else:
                keep.append((piv, price))
        unbroken = keep
        up[i] = len(highs) >= 2 and len(lows) >= 2 and highs[-1] > highs[-2] and lows[-1] > lows[-2]
        level[i] = broken if cfg.support_mode == "broken_high" else (lows[-1] if lows else np.nan)
        target[i] = highs[-1] if highs else np.nan
    half = atr * cfg.htf_zone_atr
    return pd.DataFrame({"Up": up, "Level": level, "ZoneLo": level - half, "ZoneHi": level + half,
                         "Target": target, "ATR": atr}, index=htf.index)


def map_htf_to_ltf(ltf, htf_ctx, labels):
    """上位足の状態を1本ずらし（確定した上位足の分だけ）、下位足に割り当てる"""
    shifted = htf_ctx.shift(1)
    shifted["Up"] = shifted["Up"].fillna(False).astype(bool)
    return shifted.reindex(labels).set_axis(ltf.index)


# ---------------------------------------------------------------------------
# シグナル（状態遷移）と約定
# ---------------------------------------------------------------------------
IDLE, WAIT_CHOCH, WAIT_RR, WAIT_TRIGGER = range(4)


class _Exec:
    """エントリー後の SL/TP 判定。exec_1min なら1分足、そうでなければ下位足で行う"""

    def __init__(self, ltf, df_1min, cfg):
        self.ltf_n = len(ltf)
        if df_1min is not None and cfg.exec_1min:
            lab = get_signal_bar_label(df_1min.index, cfg.session_start_hour, cfg.ltf_hours)
            pos = pd.Series(np.arange(len(df_1min))).groupby(np.asarray(lab)).min()
            self.first = pos.reindex(ltf.index).to_numpy()      # 下位足 → 最初の1分足の位置
            self.bar_of = pd.Index(ltf.index).get_indexer(lab)   # 1分足 → 下位足の位置
            src = df_1min
        else:
            self.first = np.arange(self.ltf_n)
            self.bar_of = np.arange(self.ltf_n)
            src = ltf
        self.t = src.index
        self.o, self.h, self.l, self.c = (src[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))

    def run(self, k_entry, sl, tp):
        """下位足 k_entry の始値で買い、決済まで進める → (約定位置, エントリー価格, 決済位置, 決済価格, 理由)"""
        i0 = int(self.first[k_entry])
        entry = self.o[i0]
        if not entry > sl:
            return None
        n = len(self.o)
        i, step = i0, 4096
        while i < n:
            j = min(n, i + step)
            hit = (self.l[i:j] <= sl) | (self.h[i:j] >= tp)
            if hit.any():
                x = i + int(np.argmax(hit))
                if self.l[x] <= sl:                              # 同じ足で両方に触れたら損切りを優先
                    return i0, entry, x, min(self.o[x], sl) if x > i0 else sl, "SL"
                return i0, entry, x, max(self.o[x], tp) if x > i0 else tp, "TP"
            i, step = j, step * 2
        return i0, entry, n - 1, self.c[-1], "END"


def run_strategy(ltf, ctx, cfg, df_1min=None, commission=0.0, trade_from=None):
    """
    下位足（Open/High/Low/Close）と、下位足に割り当て済みの上位足の状態（map_htf_to_ltf）から、トレード一覧を返す。
    trade_from: この位置より前の下位足ではセットアップを始めない（助走期間）
    """
    o, h, l, c = (ltf[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    m = len(ltf)
    n = cfg.swing_n_ltf
    atr = calculate_atr_ema(h, l, c, cfg.atr_len, cfg.price_decimals)
    sh_piv, sl_piv = (list(map(int, a)) for a in fractal_pivots(h, l, n))
    up = ctx["Up"].to_numpy(bool)
    zlo, zhi, tgt = (ctx[k].to_numpy(float) for k in ("ZoneLo", "ZoneHi", "Target"))
    ex = _Exec(ltf, df_1min, cfg)
    trades = []
    funnel = dict.fromkeys(["開始", "転換(CHoCH)", "RR接触", "エントリー", "取消:ゾーン割れ", "取消:否定ライン割れ",
                            "取消:期限切れ"], 0)
    state, k = IDLE, max(int(trade_from or 0), cfg.atr_len)
    s = {}
    while k < m - 1:
        if state == IDLE:
            if up[k] and l[k] <= zhi[k] and c[k] >= zlo[k]:
                s = dict(start=k, stage=k, bottom=l[k], bpos=k, zone_lo=zlo[k], zone_hi=zhi[k], target=tgt[k])
                state = WAIT_CHOCH
                funnel["開始"] += 1
        elif state == WAIT_CHOCH:
            if l[k] < s["bottom"]:
                s["bottom"], s["bpos"] = l[k], k
            if c[k] < s["zone_lo"] or k - s["stage"] > cfg.setup_expiry:
                funnel["取消:ゾーン割れ" if c[k] < s["zone_lo"] else "取消:期限切れ"] += 1
                state = IDLE
            else:
                # 底より前にできた確定済みスイング高値の直近2つ（H_a > H_b）。底の後の高値（H_c）は対象外
                cut = bisect.bisect_left(sh_piv, min(s["bpos"], k - n + 1))
                if cut >= 2:
                    pa, pb = sh_piv[cut - 2], sh_piv[cut - 1]
                    if h[pa] > h[pb] and c[k] > h[pb]:
                        lo_i = bisect.bisect_right(sl_piv, s["bpos"])
                        hi_i = bisect.bisect_right(sl_piv, k - n)
                        hl = sl_piv[hi_i - 1] if hi_i > lo_i else None
                        s.update(choch=k, ha=h[pa], ha_pos=pa, rr=h[pb], rr_pos=pb, stage=k, away=False,
                                 inval=l[hl] if hl is not None else s["bottom"], hl_pos=hl)
                        state = WAIT_RR
                        funnel["転換(CHoCH)"] += 1
        elif state == WAIT_RR:
            if c[k] < s["inval"] or k - s["stage"] > cfg.setup_expiry:
                funnel["取消:否定ライン割れ" if c[k] < s["inval"] else "取消:期限切れ"] += 1
                state = IDLE
            elif not s["away"]:
                s["away"] = h[k] >= s["rr"] + atr[k] * cfg.rr_away_atr
            elif l[k] <= s["rr"] + atr[k] * cfg.rr_zone_atr:
                s.update(contact=k, touch_low=l[k], stage=k)
                state = WAIT_TRIGGER
                funnel["RR接触"] += 1
        elif state == WAIT_TRIGGER:
            s["touch_low"] = min(s["touch_low"], l[k])
            if c[k] < s["inval"] or k - s["stage"] > cfg.setup_expiry:
                funnel["取消:否定ライン割れ" if c[k] < s["inval"] else "取消:期限切れ"] += 1
                state = IDLE
            elif k >= cfg.trigger_n and c[k] > h[k - cfg.trigger_n:k].max():
                sl = s["touch_low"] - atr[k] * cfg.sl_buffer_atr
                entry = ex.o[int(ex.first[k + 1])]       # 次の足の始値
                state = IDLE
                if entry > sl:
                    risk = entry - sl
                    tp = entry + risk * cfg.rr_target if cfg.rr_target is not None else s["target"]
                    if tp > entry:
                        funnel["エントリー"] += 1
                        i0, entry, xi, xp, why = ex.run(k + 1, sl, tp)
                        r_gross = (xp - entry) / risk
                        r = r_gross - (abs(entry) + abs(xp)) * commission / risk
                        k_exit = int(ex.bar_of[xi])
                        trades.append(dict(
                            entry_time=ex.t[i0], entry=entry, sl=sl, tp=tp, exit_time=ex.t[xi], exit=xp,
                            exit_pos=k_exit, exit_reason=why, R=r, R_gross=r_gross, rr_level=s["rr"],
                            swing_low=s["touch_low"], contact_price=l[s["contact"]], h_a=s["ha"], bottom=s["bottom"], inval=s["inval"],
                            zone_lo=s["zone_lo"], zone_hi=s["zone_hi"], htf_target=s["target"],
                            setup_pos=s["start"], bottom_pos=s["bpos"], ha_pos=s["ha_pos"], rr_pos=s["rr_pos"],
                            hl_pos=s["hl_pos"], choch_pos=s["choch"], contact_pos=s["contact"],
                            signal_pos=k, entry_pos=k + 1,
                        ))
                        k = max(k_exit, k)             # 保有中は新しいシグナルを出さない
        k += 1
    out = pd.DataFrame(trades)
    if len(out):
        out["side"] = "long"
    out.attrs["funnel"] = funnel                       # 各段階を通過した数・取り消しの理由
    return out


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------
def summarize(r):
    r = np.asarray(r, float)
    curve = np.concatenate([[0.0], np.cumsum(r)])
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "トレード数": int(len(r)),
        "勝率 [%]": float((r > 0).mean() * 100) if len(r) else float("nan"),
        "平均R": float(r.mean()) if len(r) else float("nan"),
        "合計R": float(r.sum()),
        "最大DD [R]": float((np.maximum.accumulate(curve) - curve).max()),
        "PF": float(wins / losses) if losses > 0 else float("inf"),
    }


def to_chart_trades(trades, cfg):
    """dow_swing_chart.make_chart() に渡せる形（backtesting.py の _trades と同じ列名）"""
    risk_cash = cfg.cash * cfg.risk_pct
    size = (risk_cash / (trades["entry"] - trades["sl"]).abs()).round().astype(int).clip(lower=1)
    size = size.where(trades["side"] == "long", -size)
    return pd.DataFrame({
        "Size": size, "EntryTime": trades["entry_time"], "ExitTime": trades["exit_time"],
        "EntryPrice": trades["entry"], "ExitPrice": trades["exit"], "SL": trades["sl"], "TP": trades["tp"],
        "PnL": trades["R"] * risk_cash, "ReturnPct": trades["R"] * cfg.risk_pct,
    })


# ---------------------------------------------------------------------------
# セットアップの確認用チャート（1トレード1枚の SVG）
# ---------------------------------------------------------------------------
def setup_svg(ltf, tr, cfg, pad_before=25, pad_after=12, width=720, height=300):
    a = max(0, int(min(tr["setup_pos"], tr["ha_pos"])) - pad_before)
    b = min(len(ltf) - 1, int(tr["exit_pos"]) + pad_after)
    if b - a > 260:                                   # 長い保有は決済側を切る
        b = a + 260
    seg = ltf.iloc[a:b + 1]
    short = tr["side"] == "short"
    rng = seg["High"].max() - seg["Low"].min()
    tp_in = min(max(tr["tp"], seg["Low"].min() - 0.3 * rng), seg["High"].max() + 0.3 * rng)   # 遠い TP は枠の端に寄せる
    lo = min(seg["Low"].min(), tr["sl"], tr["zone_lo"], tp_in)
    hi = max(seg["High"].max(), tr["sl"], tr["zone_hi"], tp_in)
    padv = (hi - lo) * 0.04
    lo, hi = lo - padv, hi + padv
    L, R, T, B = 8, 64, 8, 22
    W, H = width - L - R, height - T - B
    cw = W / len(seg)
    X = lambda i: L + (i - a + 0.5) * cw
    Y = lambda p: T + (hi - p) / (hi - lo) * H
    d = cfg.price_decimals
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="セットアップ">']
    out.append(f'<rect class="zone" x="{L}" width="{W}" y="{Y(tr["zone_hi"]):.1f}" '
               f'height="{max(1, Y(tr["zone_lo"]) - Y(tr["zone_hi"])):.1f}"/>')
    for i, (_, row) in enumerate(seg.iterrows(), start=a):
        up = row["Close"] >= row["Open"]
        x = X(i)
        y1, y2 = Y(max(row["Open"], row["Close"])), Y(min(row["Open"], row["Close"]))
        out.append(f'<line class="wick" x1="{x:.1f}" x2="{x:.1f}" y1="{Y(row["High"]):.1f}" y2="{Y(row["Low"]):.1f}"/>'
                   f'<rect class="{"cu" if up else "cd"}" x="{x - cw * 0.35:.1f}" width="{max(cw * 0.7, 0.8):.1f}" '
                   f'y="{y1:.1f}" height="{max(y2 - y1, 0.6):.1f}"/>')

    def hline(cls, p, x0, x1, label):
        if not lo <= p <= hi:
            return
        x0, x1 = X(max(x0, a)) - cw / 2, X(min(x1, b)) + cw / 2
        out.append(f'<line class="{cls}" x1="{x0:.1f}" x2="{x1:.1f}" y1="{Y(p):.1f}" y2="{Y(p):.1f}"/>'
                   f'<text class="lb {cls}t" x="{L + W + 4}" y="{Y(p) + 4:.1f}">{label}</text>')

    nm = dict(a="L_a", b="L_b", bottom="天井", hl="LH") if short else dict(a="H_a", b="H_b", bottom="底", hl="HL")
    hline("rr", tr["rr_level"], tr["rr_pos"], tr["entry_pos"], nm["b"])
    hline("inv", tr["inval"], tr["choch_pos"], tr["entry_pos"], "否定")
    hline("sl", tr["sl"], tr["entry_pos"], tr["exit_pos"], "SL")
    hline("tp", tr["tp"], tr["entry_pos"], tr["exit_pos"], "TP")
    hline("en", tr["entry"], tr["entry_pos"], tr["exit_pos"], "IN")
    up_, dn_ = (16, -8) if short else (-8, 16)       # ラベルを置く向き（ロングは高値が上・安値が下）
    pts = [("ha", tr["ha_pos"], tr["h_a"], nm["a"], dn_), ("ha", tr["rr_pos"], tr["rr_level"], nm["b"], dn_),
           ("bt", tr["bottom_pos"], tr["bottom"], nm["bottom"], up_), ("ct", tr["contact_pos"],
           tr["contact_price"], "接触", up_)]
    if not pd.isna(tr["hl_pos"]):
        pts.append(("hl", tr["hl_pos"], tr["inval"], nm["hl"], up_))
    for cls, i, p, label, dy in pts:
        if a <= i <= b:
            out.append(f'<circle class="{cls}" cx="{X(i):.1f}" cy="{Y(p):.1f}" r="4"/>'
                       f'<text class="pt" x="{X(i):.1f}" y="{Y(p) + dy:.1f}" text-anchor="middle">{label}</text>')
    for cls, i, sym in (("ch", tr["choch_pos"], "CHoCH"), ("en", tr["entry_pos"], "IN"), ("ex", tr["exit_pos"], "OUT")):
        if a <= i <= b:
            out.append(f'<line class="vt {cls}" x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{T}" y2="{T + H}"/>'
                       f'<text class="pt" x="{X(i):.1f}" y="{height - 6}" text-anchor="middle">{sym}</text>')
    for p in (lo + padv, hi - padv):
        out.append(f'<text class="ax" x="{L + W + 4}" y="{Y(p) + 4:.1f}">{p:.{d}f}</text>')
    out.append("</svg>")
    return "".join(out)


SETUP_CSS = """
:root{--bg:#f3f5f7;--surface:#fff;--line:#dde2e8;--fg:#17202b;--muted:#5e6b7a;--pos:#18865a;--neg:#c8413a;--zone:#f0a03a;--rr:#d63b3b;--acc:#2a5bd7;--grn:#1f9d5a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f141a;--surface:#171e26;--line:#2a333e;--fg:#e3e8ee;--muted:#93a0ae;--pos:#3ccb8b;--neg:#f0726a;--zone:#e0a050;--rr:#f06a6a;--acc:#79a2ff;--grn:#3ccb8b;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f141a;--surface:#171e26;--line:#2a333e;--fg:#e3e8ee;--muted:#93a0ae;--pos:#3ccb8b;--neg:#f0726a;--zone:#e0a050;--rr:#f06a6a;--acc:#79a2ff;--grn:#3ccb8b;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 "Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:20px 16px 48px;display:grid;gap:18px}h1{font-size:1.3rem;margin:0}
.sub,.meta{color:var(--muted);font-size:.82rem}.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px;min-width:0}
.card h2{font-size:.95rem;margin:0 0 4px;font-family:"IBM Plex Mono",ui-monospace,monospace}.pos{color:var(--pos)}.neg{color:var(--neg)}
svg{width:100%;height:auto;display:block}.wick{stroke:var(--muted);stroke-width:1}.cu{fill:var(--pos)}.cd{fill:var(--neg)}
.zone{fill:var(--zone);opacity:.18}.rr{stroke:var(--rr);stroke-width:1.5}.inv{stroke:var(--muted);stroke-dasharray:3 3}
.sl{stroke:var(--neg);stroke-dasharray:5 3}.tp{stroke:var(--pos);stroke-dasharray:5 3}.en{stroke:var(--acc)}
.vt{stroke-width:1;stroke-dasharray:2 3}.vt.ch{stroke:var(--rr)}.vt.en{stroke:var(--acc)}.vt.ex{stroke:var(--muted)}
.ha{fill:var(--rr)}.bt{fill:var(--zone)}.ct{fill:none;stroke:var(--grn);stroke-width:2}.hl{fill:var(--muted)}
.pt,.lb,.ax{font:10px "IBM Plex Mono",ui-monospace,monospace;fill:var(--muted)}.rrt{fill:var(--rr)}.slt{fill:var(--neg)}.tpt{fill:var(--pos)}.ent{fill:var(--acc)}
a{color:var(--acc)}.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.78rem;color:var(--muted)}
"""


def write_setup_page(path, ltf, trades, cfg, title):
    cards = []
    for k, tr in trades.reset_index(drop=True).iterrows():
        cls = "pos" if tr["R"] > 0 else "neg"
        cards.append(
            f'<div class="card" id="t{k + 1}"><h2>#{k + 1} {"売り" if tr["side"] == "short" else "買い"} '
            f'<span class="{cls}">{tr["R"]:+.2f}R</span> {tr["exit_reason"]}</h2><p class="meta">エントリー {tr["entry_time"]:%Y-%m-%d %H:%M} @ '
            f'{tr["entry"]:.{cfg.price_decimals}f} ／ 決済 {tr["exit_time"]:%Y-%m-%d %H:%M} @ '
            f'{tr["exit"]:.{cfg.price_decimals}f}</p>{setup_svg(ltf, tr, cfg)}</div>')
    legend = ('<div class="legend"><span>オレンジ帯: 上位足サポート（売りはレジスタンス）ゾーン</span><span>赤: H_a / H_b（RR水準）。売りは上下反転</span>'
              '<span>オレンジ点: 底</span><span>緑の丸: 接触</span><span>灰の点線: 否定ライン（HL）</span>'
              '<span>青: エントリー / 赤緑の破線: SL・TP</span><span>縦線: CHoCH・IN・OUT</span></div>')
    html = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>セットアップ一覧</title>'
            f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;700'
            f'&family=IBM+Plex+Mono:wght@400;600&display=swap"><style>{SETUP_CSS}</style></head><body><div class="wrap">'
            f'<div><h1>{escape(title)}</h1><p class="sub">下位足（{cfg.ltf_hours}時間足・NY時間）のローソク足。'
            f'検出したセットアップが目視の形と合っているかを確かめる用。</p></div>{legend}{"".join(cards)}'
            f'<p class="sub"><a href="report.html" onclick="history.back();return false">← 結果に戻る</a></p></div></body></html>')
    Path(path).write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------
def prepare(df_1min, cfg, side="long"):
    """1分足 → 下位足・上位足・上位足の状態（下位足に割り当て済み）。side="short" は価格を反転した世界で作る"""
    if side == "short":
        df_1min = mirror(df_1min)
    ltf = resample_to_signal_bars(df_1min, cfg.session_start_hour, cfg.ltf_hours)
    labels = htf_labels(ltf.index, cfg.htf_rule, cfg.session_start_hour)
    htf = resample_bars(ltf, labels)
    ctx_htf = htf_context(htf, cfg)
    return ltf, htf, ctx_htf, map_htf_to_ltf(ltf, ctx_htf, labels), df_1min


def main(cfg=CFG):
    t0 = time.time()
    start_ts = pd.Timestamp(cfg.start_date) if cfg.start_date else None
    end_ts = pd.Timestamp(cfg.end_date) if cfg.end_date else None
    lo = hi = None
    if start_ts is not None:
        p = start_ts.to_period("M") - WARMUP_MONTHS
        lo = p.year * 100 + p.month
    if end_ts is not None:
        p = end_ts.to_period("M") + 1
        hi = p.year * 100 + p.month
    df_1min, spread = load_gmo_click_1min_data(
        cfg.data_root / cfg.pair, price_side="mid", price_decimals=cfg.price_decimals,
        month_range=(lo, hi), spread_period=(start_ts, end_ts), session_start_hour=cfg.session_start_hour)
    day = get_trading_day_label(df_1min.index, cfg.session_start_hour)
    keep = np.ones(len(df_1min), bool) if end_ts is None else np.asarray(day <= end_ts)
    df_1min = df_1min[keep]
    print(f"⏱ 読み込み {time.time() - t0:.1f} 秒")

    t1 = time.time()
    ltf_day = None
    parts, funnels = [], {}
    for side in cfg.sides:
        ltf_s, htf_s, ctx_htf, ctx, src = prepare(df_1min, cfg, side)
        if ltf_day is None:
            ltf_day = (ltf_s.index - pd.Timedelta(hours=cfg.session_start_hour)).normalize() + pd.Timedelta(days=1)
            trade_from = int(np.argmax(ltf_day >= start_ts)) if start_ts is not None else 0
        t = run_strategy(ltf_s, ctx, cfg, src, commission=spread / 2, trade_from=trade_from)
        funnels[side] = t.attrs.get("funnel")
        if len(t) and side == "short":
            t = unmirror_trades(t)
        parts.append(t)
    ltf = resample_to_signal_bars(df_1min, cfg.session_start_hour, cfg.ltf_hours)   # 表示用（実際の価格）
    trades = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) \
        else pd.DataFrame()
    if len(trades):
        trades = trades.sort_values("entry_time", kind="stable").reset_index(drop=True)
    print(f"⏱ 検証 {time.time() - t1:.1f} 秒（下位足 {len(ltf):,} 本 / 上位足 {len(htf_s):,} 本）")

    stats = summarize(trades["R"] if len(trades) else [])
    stats_side = {sd: summarize(trades.loc[trades["side"] == sd, "R"] if len(trades) else []) for sd in cfg.sides}
    for sd, f in funnels.items():
        print(f"  段階ごとの件数（{sd}）:", f)
    for name, st in (("全体", stats), *stats_side.items()):
        print(f"  [{name}] " + " / ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in st.items()))

    label = RUN_LABEL or f"{cfg.pair}_{cfg.htf_rule}x{cfg.ltf_hours}h_{'-'.join(cfg.sides)}"
    out_dir = RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(cfg).items()}
    (out_dir / "config.json").write_text(json.dumps(dict(cfg=cfg_json, avg_relative_spread=spread,
                                                         stats=stats, stats_by_side=stats_side, funnel=funnels),
                                                    ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    (out_dir / "code").mkdir(exist_ok=True)
    for src in (Path(__file__), Path(__file__).with_name("mtf_report.py")):
        shutil.copy2(src, out_dir / "code" / src.name)
    if len(trades):
        trades.to_csv(out_dir / f"trades_{cfg.pair}.csv", index=False, encoding="utf-8-sig")
        write_setup_page(out_dir / "setups.html", ltf, trades, cfg,
                         f"{cfg.pair} セットアップ一覧（上位足 {cfg.htf_rule} × 下位足 {cfg.ltf_hours}h）")
        chart_file = f"dow_trade_chart_{label}.html"
        try:
            from dow_swing_chart import make_chart
            make_chart(df_1min, out_dir / chart_file,
                       f"{cfg.pair} {cfg.start_date} ~ {cfg.end_date} / 上位足押し目×下位足転換",
                       cfg.start_date, cfg.end_date, trades=to_chart_trades(trades, cfg), weekly_trend=None,
                       mode="mtf_pullback", risk=cfg.cash * cfg.risk_pct, price_decimals=cfg.price_decimals,
                       ma_periods=(20, 50, 120))
        except Exception as e:  # チャートの失敗で検証結果を失わない
            print(f"⚠️ チャートを作れませんでした: {e}")
        from mtf_report import write_report
        write_report(out_dir / "report.html", trades, stats, stats_side, funnels, cfg, spread, chart_file,
                     f"{cfg.start_date} ~ {cfg.end_date}")
    print(f"💾 出力先: {out_dir.resolve()}（合計 {time.time() - t0:.1f} 秒）")
    return dict(trades=trades, stats=stats, stats_by_side=stats_side, funnels=funnels, ltf=ltf, out_dir=out_dir,
                spread=spread, df_1min=df_1min)


if __name__ == "__main__":
    main()
