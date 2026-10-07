"""
スイングブレイクアウト（fast_results の検証結果）の各トレードに、エントリー直前で確定していた
「トレンドの方向・強さ」の指標を付け、指標の値ごとに成績が変わるかを調べる。

- 戦略は再実行しない。fast_results/<RUN>/trades_all.csv の取引に指標を付けて集計するだけ。
  （フィルターで取引を除外しても、残った取引は変わらない前提。ポジション保有中は次の注文を置かないため、
   実際にフィルターを入れると「除外した取引の代わりに入る取引」が生じる点は未反映）
- 指標は cache/h4_<ペア>.pkl（build_h4_cache.py）から 4時間足・日足・週足を作って計算する。
  どれも「エントリー時刻より前に終わった足」の値だけを使う（ルックアヘッドなし）。
- 方向のある指標は「取引方向に揃えた値」（買いなら +、売りなら符号反転）にする。
- 期間を分けて評価する: IS（2021-2023）で区切り値を決め、OOS（2024-2025）で確かめる。

使い方: python trend_filter_study.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dow_trend import DowConfig, compute_dow_trend
from main_4H_fixedSL import signal_bar_trading_day

RUNS = {
    "fixed": "fast_results/20261006_2301_fixed_sl1.5_tp2.5",
    "trail": "fast_results/20261006_2301_trailH1w5_atr_sl1.5",
}
CACHE = Path("cache")
IS_END = pd.Timestamp("2023-12-31")
OUT_ROOT = Path("filter_results")


# ---------------------------------------------------------------- 指標
def adx_di(df, period=14):
    """ADX と +DI - -DI（ワイルダー平滑化）"""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), df.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), df.index)
    ew = lambda x: x.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    atr = ew(tr)
    pdi, ndi = 100 * ew(pdm) / atr, 100 * ew(ndm) / atr
    adx = ew(100 * (pdi - ndi).abs() / (pdi + ndi))
    return adx, pdi - ndi


def er_signed(c, n):
    """効率比（符号付き）: (C - C[n]) / Σ|ΔC|。+1 に近いほど一直線の上昇"""
    return (c - c.shift(n)) / c.diff().abs().rolling(n).sum()


def chop(df, n=14):
    """チョピネスインデックス。高いほどレンジ（61.8 以上）、低いほどトレンド（38.2 以下）"""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    rng = h.rolling(n).max() - l.rolling(n).min()
    return 100 * np.log10(tr.rolling(n).sum() / rng) / np.log10(n)


def linreg_r2_signed(c, n):
    """終値 n 本の線形回帰の決定係数 R² に傾きの符号を付けたもの（-1〜+1）"""
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()

    def f(y):
        ym = y.mean()
        sxy = ((x - xm) * (y - ym)).sum()
        syy = ((y - ym) ** 2).sum()
        if syy == 0:
            return 0.0
        return np.sign(sxy) * sxy * sxy / (sxx * syy)

    return c.rolling(n).apply(f, raw=True)


def h4_features(h4):
    f = pd.DataFrame(index=h4.index)
    c = h4["Close"]
    f["H4_ER18"] = er_signed(c, 18)
    f["H4_ADX14"], f["H4_DI"] = adx_di(h4, 14)
    f["H4_CHOP14"] = chop(h4, 14)
    f["H4_R2_36"] = linreg_r2_signed(c, 36)
    f["H4_MA_ORDER"] = ma_order(c, 20, 50, 120)
    f["end"] = h4.index + pd.Timedelta(hours=4)
    return f


def ma_order(c, a, b, d):
    """+1: 短期 > 中期 > 長期（上昇の並び）、-1: その逆、0: それ以外"""
    ma1, ma2, ma3 = c.rolling(a).mean(), c.rolling(b).mean(), c.rolling(d).mean()
    o = np.where((ma1 > ma2) & (ma2 > ma3), 1, np.where((ma1 < ma2) & (ma2 < ma3), -1, 0)).astype(float)
    o[ma3.isna().to_numpy()] = np.nan
    return pd.Series(o, c.index)


def daily_features(d1):
    f = pd.DataFrame(index=d1.index)
    c = d1["Close"]
    f["D1_ER10"] = er_signed(c, 10)
    f["D1_ER20"] = er_signed(c, 20)
    f["D1_ADX14"], f["D1_DI"] = adx_di(d1, 14)
    f["D1_CHOP14"] = chop(d1, 14)
    f["D1_R2_20"] = linreg_r2_signed(c, 20)
    f["D1_MA_ORDER"] = ma_order(c, 20, 50, 100)
    f["D1_ABOVE_MA100"] = np.sign(c - c.rolling(100).mean())
    f["D1_MA50_SLOPE"] = np.sign(c.rolling(50).mean().diff(5))
    hi, lo = d1["High"].rolling(20).max(), d1["Low"].rolling(20).min()
    f["D1_DONCH20"] = 2 * (c - lo) / (hi - lo) - 1          # -1（安値圏）〜 +1（高値圏）
    f["D1_ROC20"] = np.sign(c / c.shift(20) - 1)
    # 取引日 D の足は NY 17:00 に終わる
    f["end"] = d1.index + pd.Timedelta(hours=17)
    return f


def weekly_features(w1):
    f = pd.DataFrame(index=w1.index)
    c = w1["Close"]
    trend, _ = compute_dow_trend(w1, DowConfig(n=3, atr_period=14, min_swing_atr=1.0, use_wick=True))
    f["W1_DOW"] = trend.astype(float)
    f["W1_DOW_AGE"] = (trend.groupby((trend != trend.shift()).cumsum()).cumcount() + 1).astype(float)
    f["W1_ADX14"], f["W1_DI"] = adx_di(w1, 14)
    f["W1_ER13"] = er_signed(c, 13)
    f["W1_ROC13"] = np.sign(c / c.shift(13) - 1)
    f["W1_ABOVE_MA26"] = np.sign(c - c.rolling(26).mean())
    f["end"] = w1.index + pd.Timedelta(hours=17)   # 金曜 NY 17:00
    return f


def build_features(pair):
    h4 = pd.read_pickle(CACHE / f"h4_{pair}.pkl")
    td = signal_bar_trading_day(h4.index)
    agg = dict(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    d1 = h4.groupby(td).agg(**agg)
    wk = td.to_period("W-FRI").end_time.normalize()
    w1 = h4.groupby(wk).agg(**agg)
    return h4_features(h4), daily_features(d1), weekly_features(w1)


def attach(trades, feats):
    """エントリー時刻（NY 壁時計）より前に終わった最新の足の値を付ける"""
    t = trades.sort_values("t").copy()
    for f in feats:
        f = f.sort_values("end")
        t = pd.merge_asof(t, f.drop(columns=[]).reset_index(drop=True), left_on="t", right_on="end",
                          allow_exact_matches=True, direction="backward").drop(columns="end")
    return t


SIGNED = ["H4_ER18", "H4_DI", "H4_R2_36", "H4_MA_ORDER", "D1_ER10", "D1_ER20", "D1_DI", "D1_R2_20",
          "D1_MA_ORDER", "D1_ABOVE_MA100", "D1_MA50_SLOPE", "D1_DONCH20", "D1_ROC20", "W1_DOW", "W1_DI",
          "W1_ER13", "W1_ROC13", "W1_ABOVE_MA26"]
UNSIGNED = ["H4_ADX14", "H4_CHOP14", "D1_ADX14", "D1_CHOP14", "W1_ADX14", "D1_ABS_ER20", "D1_ABS_R2_20",
            "H4_ABS_ER18", "W1_ABS_ER13"]
CATEGORICAL = ["H4_MA_ORDER", "D1_MA_ORDER", "D1_ABOVE_MA100", "D1_MA50_SLOPE", "D1_ROC20", "W1_DOW",
               "W1_ROC13", "W1_ABOVE_MA26"]


def load_trades(run_dir):
    t = pd.read_csv(Path(run_dir) / "trades_all.csv", encoding="utf-8-sig")
    et = pd.to_datetime(t["EntryTime"], utc=True).dt.tz_convert("America/New_York")
    t["t"] = et.dt.tz_localize(None)
    t["dir"] = np.where(t["Size"] > 0, 1, -1)
    t["IS"] = t["t"] <= IS_END + pd.Timedelta(days=1)
    t["year"] = t["t"].dt.year
    return t


def enrich(t, feats_by_pair):
    out = []
    for pair, g in t.groupby("Pair"):
        out.append(attach(g, feats_by_pair[pair]))
    t = pd.concat(out).sort_values("t").reset_index(drop=True)
    for col in ["H4_ER18", "D1_ER20", "W1_ER13", "D1_R2_20"]:
        name = col.split("_", 1)
        t[f"{name[0]}_ABS_{name[1]}"] = t[col].abs()
    for col in SIGNED:
        t[col] = t[col] * t["dir"]                    # 取引方向に揃える
    # ダウ理論の継続週数は、取引方向に揃ったトレンドのときだけ意味を持たせる（逆行・レンジなら 0）
    t["W1_DOW_AGE_ALIGNED"] = np.where(t["W1_DOW"] > 0, t["W1_DOW_AGE"], 0.0)
    return t


# ---------------------------------------------------------------- 集計
def stat(r):
    r = pd.Series(r, dtype=float)
    if len(r) == 0:
        return dict(n=0, R=0.0, avgR=np.nan, win=np.nan, PF=np.nan)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), R=r.sum(), avgR=r.mean(), win=(r > 0).mean() * 100, PF=gp / gl if gl > 0 else np.nan)


def bucket_table(t, col, q=4):
    """全期間をまとめた分位で区切った成績（カテゴリ指標はそのまま）"""
    d = t.dropna(subset=[col])
    if col in CATEGORICAL:
        keys = d[col]
    else:
        keys = pd.qcut(d[col], q, duplicates="drop")
    rows = []
    for k, g in d.groupby(keys, observed=True):
        s = stat(g["R"])
        s_is, s_oos = stat(g.loc[g.IS, "R"]), stat(g.loc[~g.IS, "R"])
        rows.append(dict(bucket=str(k), **s, IS_avgR=s_is["avgR"], OOS_avgR=s_oos["avgR"],
                         IS_n=s_is["n"], OOS_n=s_oos["n"]))
    return pd.DataFrame(rows)


def threshold_test(t, col):
    """IS の分位（中央値・上位1/3）で区切り値を決め、IS と OOS で「区切り値以上だけ取引」の成績を見る"""
    d = t.dropna(subset=[col])
    is_ = d[d.IS]
    rows = []
    if col in CATEGORICAL:
        cands = [("> 0", 0.5)] + ([(">= 0", -0.5)] if d[col].min() < 0 else [])
    else:
        cands = [(f"IS中央値 {is_[col].quantile(.5):.3g}", is_[col].quantile(.5)),
                 (f"IS上位1/3 {is_[col].quantile(2/3):.3g}", is_[col].quantile(2 / 3))]
        if col in ("H4_CHOP14", "D1_CHOP14"):     # CHOP は低いほどトレンド
            cands = [(f"IS中央値以下 {is_[col].quantile(.5):.3g}", -is_[col].quantile(.5)),
                     (f"IS下位1/3 {is_[col].quantile(1/3):.3g}", -is_[col].quantile(1 / 3))]
    for label, th in cands:
        v = -d[col] if col in ("H4_CHOP14", "D1_CHOP14") else d[col]
        keep = v >= th
        for part, m in (("IS", d.IS), ("OOS", ~d.IS)):
            k, r = stat(d.loc[m & keep, "R"]), stat(d.loc[m & ~keep, "R"])
            rows.append(dict(ind=col, rule=label, part=part, keep_n=k["n"], keep_R=k["R"], keep_avgR=k["avgR"],
                             keep_PF=k["PF"], drop_n=r["n"], drop_R=r["R"], drop_avgR=r["avgR"]))
        # ペア別に「残した方の平均R − 除外した方の平均R」が正のペア数
        diffs = []
        for _, g in d.groupby("Pair"):
            vv = -g[col] if col in ("H4_CHOP14", "D1_CHOP14") else g[col]
            a, b = g.loc[vv >= th, "R"], g.loc[vv < th, "R"]
            if len(a) >= 5 and len(b) >= 5:
                diffs.append(a.mean() - b.mean())
        rows[-1]["pairs_better"] = f"{sum(x > 0 for x in diffs)}/{len(diffs)}"
        rows[-2]["pairs_better"] = rows[-1]["pairs_better"]
    return rows


def spearman(t, col):
    d = t.dropna(subset=[col])
    rho = d[col].rank().corr(d["R"].rank())
    rho_is = d[d.IS][col].rank().corr(d[d.IS]["R"].rank())
    rho_oos = d[~d.IS][col].rank().corr(d[~d.IS]["R"].rank())
    return dict(ind=col, n=len(d), rho=rho, rho_IS=rho_is, rho_OOS=rho_oos,
                t=rho * np.sqrt((len(d) - 2) / max(1e-9, 1 - rho ** 2)))


def main():
    pairs = sorted({p for d in RUNS.values() for p in pd.read_csv(Path(d) / "trades_all.csv")["Pair"].unique()})
    feats = {p: build_features(p) for p in pairs}
    out_dir = OUT_ROOT / f"{datetime.now():%Y%m%d_%H%M}_trend_filters"
    out_dir.mkdir(parents=True, exist_ok=True)
    indicators = SIGNED + UNSIGNED + ["W1_DOW_AGE_ALIGNED"]
    result = {}
    for mode, run in RUNS.items():
        t = enrich(load_trades(run), feats)
        t.to_csv(out_dir / f"trades_enriched_{mode}.csv", index=False, encoding="utf-8-sig")
        sp = pd.DataFrame([spearman(t, c) for c in indicators]).sort_values("t", ascending=False)
        sp.to_csv(out_dir / f"spearman_{mode}.csv", index=False, encoding="utf-8-sig")
        th = pd.DataFrame([r for c in indicators for r in threshold_test(t, c)])
        th.to_csv(out_dir / f"threshold_{mode}.csv", index=False, encoding="utf-8-sig")
        buckets = {c: bucket_table(t, c).to_dict("records") for c in indicators}
        base = {p: stat(t.loc[m, "R"]) for p, m in (("IS", t.IS), ("OOS", ~t.IS), ("ALL", t.IS | ~t.IS))}
        result[mode] = dict(base=base, spearman=sp.to_dict("records"), threshold=th.to_dict("records"),
                            buckets=buckets)
        print(f"\n===== {mode}  base IS {base['IS']['R']:+.1f}R / OOS {base['OOS']['R']:+.1f}R")
        print(sp.round(3).to_string(index=False))
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, default=float, indent=1)
    print(f"\n結果: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
