#!/usr/bin/env python3
"""
取引履歴CSVの診断スクリプト。

入力: main_4H.py が出力する取引履歴CSV（単発検証の trade_history_*.csv、
      またはグリッド検証の grid_trades/trades_SL◯_TP◯.csv）。

出力: ・端末に診断レポートを表示
      ・<out-dir>/diagnosis_report.txt  … 同じ内容のテキスト
      ・<out-dir>/trades_enriched.csv   … 各トレードに診断用の列（方向・決済理由・時間帯・
                                          ATR水準・MAE/MFE など）を付けたCSV

使い方:
    python diagnose_trades.py                        # 引数なし: カレントフォルダの最新の trade_history_*.csv
    python diagnose_trades.py sl1.5_tp2.5            # 名前の一部だけ指定（部分一致で探す）
    python diagnose_trades.py trade_history_xxx.csv  # ファイル名をそのまま指定
    python diagnose_trades.py sl1.5_tp2.5 --mae-mfe  # MAE/MFEも計算（要・元の1分足データ）
    python diagnose_trades.py sl1.5_tp2.5 --mae-mfe --data-path histData/EURUSD

    ファイルを探す場所: カレントフォルダ、その下の grid_trades/、このスクリプトのあるフォルダ
    （グリッド検証の trades_SL1.5_TP2.5.csv も「SL1.5_TP2.5」などで探せます）。
    複数見つかったときは、更新日時が最新のものを使い、使ったファイル名を表示します。

MAE/MFE を計算するときは、このファイルを main_4H.py と同じフォルダに置いてください
（main_4H.py のデータ読み込み関数を使います）。

診断の内容:
    1. 全体成績（期待値・t値・最大連敗）   2. 買い/売り別   3. 年別
    4. エントリー時間帯別（4時間足の区切り）  5. 曜日別
    6. ATR水準別   7. SwingHigh-SwingLow幅別   8. 決済理由別（SL/TP/その他）
    9. 保有時間   10. コスト感度   11. MAE/MFE（任意）

注意:
    ・損益は pips に換算して集計します（PnL ÷ (|Size| × pip_size)）。EURUSDなど pip=0.0001 の通貨ペア
      が既定です。USDJPYなら --pip-size 0.01 を指定してください。
    ・トレード数が少ない区分（n<20）は偶然の影響が大きいので「参考」と表示します。
    ・「平均R」は、損益をそのトレードのSL距離（＝1R）で割った平均です。ATR水準別のように
      SL/TP幅そのものが変わる区分は、pipsではなく平均Rと勝率で比べてください。
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"
COL_ATR = "Entry_ATR (Signal, EMA)"
COL_SH = "Entry_Active Swing High Line (Signal)"
COL_SL = "Entry_Active Swing Low Line (Signal)"
WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]
SESSION_START_HOUR = 17   # 4時間足の区切りの起点（NY時間）。main_4H.py の Config と合わせる
SIGNAL_HOURS = 4          # 判定足の長さ（時間）
MIN_N = 20                # これ未満の区分は「参考」扱い


def resolve_csv(arg):
    """
    取引履歴CSVのパスを決める。
      ・指定なし        → trade_history_*.csv のうち更新日時が最新のもの
      ・存在するパス    → そのまま使う
      ・存在しない文字列 → 名前の部分一致（*文字列*.csv）で探し、複数あれば最新のもの
    探す場所は、カレントフォルダ・その下の grid_trades/・このスクリプトのあるフォルダ。
    """
    if arg and Path(arg).is_file():
        return Path(arg)

    here = Path(__file__).resolve().parent
    dirs = []
    for d in [Path.cwd(), Path.cwd() / "grid_trades", here, here / "grid_trades"]:
        if d.is_dir() and d.resolve() not in [x.resolve() for x in dirs]:
            dirs.append(d)

    # 大文字小文字は区別しない部分一致。入力に含まれる「*」は無視する（ワイルドカード風に書いても可）
    core = Path(arg).name.replace("*", "").lower() if arg else None
    skip = {"sl_tp_grid.csv", "trades_enriched.csv"}
    found = {}
    for d in dirs:
        for f in d.iterdir():
            name = f.name.lower()
            if not (f.is_file() and name.endswith(".csv")) or name in skip:
                continue
            if (core in name) if core else name.startswith("trade_history_"):
                found[f.resolve()] = f
    if not found:
        where = "、".join(str(d) for d in dirs)
        what = f"「{arg}」を含む" if arg else "trade_history_*.csv の"
        raise SystemExit(f"❌ {what}取引履歴CSVが見つかりません。探した場所: {where}\n"
                         f"   ファイル名（またはその一部）を指定するか、CSVのあるフォルダで実行してください。")
    files = sorted(found.values(), key=lambda f: f.stat().st_mtime, reverse=True)
    if len(files) > 1:
        print(f"ℹ️ {len(files)} 件見つかったため、更新日時が最新のものを使います（他の候補: "
              + "、".join(f.name for f in files[1:4]) + ("、…" if len(files) > 4 else "") + "）")
    print(f"📄 対象ファイル: {files[0]}")
    return files[0]


class Report:
    """端末表示とテキスト保存を同時に行う。"""

    def __init__(self):
        self.lines = []

    def add(self, text=""):
        print(text)
        self.lines.append(str(text))

    def section(self, title):
        self.add("")
        self.add("=" * 72)
        self.add(title)
        self.add("=" * 72)

    def table(self, df, float_fmt="{:,.2f}"):
        with pd.option_context("display.unicode.east_asian_width", True,
                               "display.width", 250, "display.max_columns", None):
            self.add(df.to_string(float_format=lambda v: float_fmt.format(v)))


# ---------------------------------------------------------------------------
# 読み込みと診断用の列の追加
# ---------------------------------------------------------------------------
def load_trades(path, pip_size, price_tol):
    t = pd.read_csv(path, encoding="utf-8-sig")
    need = ["Size", "EntryPrice", "ExitPrice", "SL", "TP", "PnL", "Commission",
            "EntryTime", "ExitTime", COL_ATR]
    missing = [c for c in need if c not in t.columns]
    if missing:
        raise SystemExit(f"❌ 取引履歴CSVに必要な列がありません: {missing}\n"
                         f"   main_4H.py が出力した取引履歴CSVを指定してください。")
    if t.empty:
        raise SystemExit("❌ 取引履歴CSVにトレードが1件もありません。")

    # 時刻は「-05:00」「-04:00」が混在した文字列なので、一度UTCにしてからNY時間に戻す
    t["EntryTime"] = pd.to_datetime(t["EntryTime"], utc=True).dt.tz_convert(NY)
    t["ExitTime"] = pd.to_datetime(t["ExitTime"], utc=True).dt.tz_convert(NY)
    t = t.sort_values("EntryTime").reset_index(drop=True)

    is_long = (t["Size"] > 0).to_numpy()
    unit = t["Size"].abs() * pip_size

    t["方向"] = np.where(is_long, "買い", "売り")
    t["損益pips"] = t["PnL"] / unit                      # コスト控除後
    t["コストpips"] = t["Commission"] / unit
    t["ATRpips"] = t[COL_ATR] / pip_size
    t["SL距離pips"] = (t["EntryPrice"] - t["SL"]).abs() / pip_size
    # R倍数: 損益を「そのトレードのSL距離（＝1R）」で割った値。SL/TP幅がATRに比例して変わる影響を
    # 取り除いて比較するための指標（pips集計だと、ATRが大きいトレードほど損益が大きく見えるため）
    t["R"] = t["損益pips"] / t["SL距離pips"]
    t["SL距離/ATR"] = (t["EntryPrice"] - t["SL"]).abs() / t[COL_ATR]
    t["TP距離/ATR"] = (t["TP"] - t["EntryPrice"]).abs() / t[COL_ATR]

    # 決済理由: 決済価格がSL/TPに到達（ギャップで超えた場合も含む）していればSL/TP、それ以外は「その他」
    ex, sl, tp = t["ExitPrice"].to_numpy(), t["SL"].to_numpy(), t["TP"].to_numpy()
    sl_hit = np.where(is_long, ex <= sl + price_tol, ex >= sl - price_tol)
    tp_hit = np.where(is_long, ex >= tp - price_tol, ex <= tp + price_tol)
    t["決済理由"] = np.select([sl_hit, tp_hit], ["SL", "TP"], default="その他")

    # エントリー時間帯: 4時間足の区切り（NY 17,21,1,5,9,13時 起点）で分類
    h = t["EntryTime"].dt.hour
    start = (((h - SESSION_START_HOUR) % 24) // SIGNAL_HOURS * SIGNAL_HOURS + SESSION_START_HOUR) % 24
    order = [(SESSION_START_HOUR + k * SIGNAL_HOURS) % 24 for k in range(24 // SIGNAL_HOURS)]
    labels = [f"{s:02d}-{(s + SIGNAL_HOURS) % 24:02d}時" for s in order]
    t["時間帯(NY)"] = pd.Categorical(start.map(dict(zip(order, labels))), categories=labels, ordered=True)

    t["曜日"] = pd.Categorical(t["EntryTime"].dt.dayofweek.map(dict(enumerate(WEEKDAY_JA))),
                              categories=WEEKDAY_JA, ordered=True)
    t["年"] = t["EntryTime"].dt.year
    t["保有時間h"] = (t["ExitTime"] - t["EntryTime"]).dt.total_seconds() / 3600

    if COL_SH in t.columns and COL_SL in t.columns:
        t["SwingHigh-Low幅/ATR"] = (t[COL_SH] - t[COL_SL]) / t[COL_ATR]
    return t


# ---------------------------------------------------------------------------
# 集計の共通処理
# ---------------------------------------------------------------------------
def stats_row(g):
    p = g["損益pips"]
    wins, losses = p[p > 0], p[p < 0]
    return pd.Series({
        "トレード数": len(g),
        "勝率[%]": (p > 0).mean() * 100,
        "合計pips": p.sum(),
        "平均pips": p.mean(),
        "平均R": g["R"].mean(),
        "平均利益pips": wins.mean() if len(wins) else np.nan,
        "平均損失pips": losses.mean() if len(losses) else np.nan,
        "PF": wins.sum() / abs(losses.sum()) if len(losses) else np.nan,
    })


def group_table(t, key):
    rows = {}
    for k, g in t.groupby(key, observed=True):
        row = stats_row(g)
        row["備考"] = f"参考(n<{MIN_N})" if len(g) < MIN_N else ""
        rows[k] = row
    out = pd.DataFrame(rows).T
    for c in out.columns[:-1]:
        out[c] = pd.to_numeric(out[c])
    out["トレード数"] = out["トレード数"].astype(int)
    return out


def tercile_label(series, unit=""):
    """3分位（小・中・大）に分け、境界値付きのラベルを返す。"""
    q1, q2 = series.quantile([1 / 3, 2 / 3])
    labels = [f"小(<{q1:.2f}{unit})", f"中({q1:.2f}～{q2:.2f}{unit})", f"大(≥{q2:.2f}{unit})"]
    return pd.Categorical(np.select([series < q1, series < q2], labels[:2], default=labels[2]),
                          categories=labels, ordered=True)


def t_stat(p):
    return p.mean() / (p.std(ddof=1) / np.sqrt(len(p))) if len(p) > 1 and p.std(ddof=1) > 0 else np.nan


def max_consecutive_losses(t):
    run = best = 0
    for v in t.sort_values(["ExitTime", "EntryTime"])["PnL"]:
        run = run + 1 if v < 0 else 0
        best = max(best, run)
    return best


# ---------------------------------------------------------------------------
# MAE / MFE
# ---------------------------------------------------------------------------
def add_mae_mfe(t, df_1min):
    """
    各トレードの保有中の1分足から、最大含み益（MFE）と最大含み損（MAE）を計算する。

    ・エントリーした分足は、エントリー前の値動きが混ざるため含めない（次の分足から）。
    ・決済した分足は含める。誤差は最大で1分足分。
    """
    idx = df_1min.index
    highs, lows = df_1min["High"].to_numpy(), df_1min["Low"].to_numpy()
    mfe, mae = [], []
    for r in t.itertuples():
        i_start = idx.searchsorted(r.EntryTime, side="left") + 1
        i_end = idx.searchsorted(r.ExitTime, side="right")
        if i_end <= 0 or i_start > len(idx):
            mfe.append(np.nan); mae.append(np.nan); continue
        i_start = min(i_start, i_end - 1)
        seg_h, seg_l = highs[i_start:i_end], lows[i_start:i_end]
        if len(seg_h) == 0:
            mfe.append(np.nan); mae.append(np.nan); continue
        if r.Size > 0:
            mfe.append(max(seg_h.max() - r.EntryPrice, 0.0))
            mae.append(max(r.EntryPrice - seg_l.min(), 0.0))
        else:
            mfe.append(max(r.EntryPrice - seg_l.min(), 0.0))
            mae.append(max(seg_h.max() - r.EntryPrice, 0.0))
    t["MFE_ATR"] = np.array(mfe) / t[COL_ATR]
    t["MAE_ATR"] = np.array(mae) / t[COL_ATR]
    t["MAE/SL距離"] = t["MAE_ATR"] / t["SL距離/ATR"]
    return t


def load_price_data(data_path):
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        m = importlib.import_module("main_4H")
    except ImportError:
        raise SystemExit("❌ main_4H.py が見つかりません。このスクリプトを main_4H.py と同じフォルダに置いてください。")
    path = data_path or m.CFG.data_path
    print(f"📥 1分足データを読み込みます: {path}")
    df_1min, _ = m.load_gmo_click_1min_data(path, price_side=m.CFG.price_side,
                                            price_decimals=m.CFG.price_decimals)
    return df_1min


# ---------------------------------------------------------------------------
# レポート
# ---------------------------------------------------------------------------
def build_report(t, rep, source_name):
    p = t["損益pips"]
    wins, losses = p[p > 0], p[p < 0]

    rep.section("0. 前提")
    rep.add(f"入力ファイル: {source_name}")
    rep.add(f"期間: {t['EntryTime'].min():%Y-%m-%d} ～ {t['ExitTime'].max():%Y-%m-%d}（NY時間）  /  トレード数: {len(t)}")
    rep.add(f"SL距離 ÷ ATR の中央値: {t['SL距離/ATR'].median():.2f}  /  TP距離 ÷ ATR の中央値: {t['TP距離/ATR'].median():.2f}"
            f"  （＝この検証のSL倍率・TP倍率）")
    rep.add(f"エントリー時ATR: 中央値 {t['ATRpips'].median():.1f} pips（最小 {t['ATRpips'].min():.1f} / 最大 {t['ATRpips'].max():.1f}）")

    rep.section("1. 全体成績")
    gp, gl = t.loc[t["PnL"] > 0, "PnL"].sum(), t.loc[t["PnL"] < 0, "PnL"].sum()
    overall = pd.Series({
        "トレード数": len(t),
        "勝率[%]": (p > 0).mean() * 100,
        "総利益[$]": gp,
        "総損失[$]": gl,
        "純損益[$]": gp + gl,
        "プロフィットファクター": gp / abs(gl) if gl < 0 else np.nan,
        "平均利益[pips]": wins.mean(),
        "平均損失[pips]": losses.mean(),
        "ペイオフレシオ": wins.mean() / abs(losses.mean()) if len(losses) else np.nan,
        "1トレード期待値[pips]": p.mean(),
        "1トレード期待値[R]（SL距離＝1R）": t["R"].mean(),
        "期待値の標準誤差[pips]": p.std(ddof=1) / np.sqrt(len(p)),
        "t値（目安: 2以上で偶然とは言いにくい）": t_stat(p),
        "最大連敗数": max_consecutive_losses(t),
        "想定コスト[pips/往復・平均]": t["コストpips"].mean(),
    })
    rep.table(overall.to_frame("値"), float_fmt="{:,.2f}")

    rep.section("2. 買い/売り別")
    rep.table(group_table(t, "方向"))

    rep.section("3. 年別（エントリー年）")
    rep.table(group_table(t, "年"))

    rep.section(f"4. エントリー時間帯別（NY時間・{SIGNAL_HOURS}時間足の区切り）")
    rep.table(group_table(t, "時間帯(NY)"))

    rep.section("5. 曜日別（エントリー時刻のNY曜日。日曜は17時以降の取引）")
    rep.table(group_table(t, "曜日"))

    rep.section("6. ATR水準別（エントリー時の判定足ATR・3分位）")
    t["ATR水準"] = tercile_label(t["ATRpips"], "pips")
    rep.table(group_table(t, "ATR水準"))

    if "SwingHigh-Low幅/ATR" in t.columns:
        rep.section("7. SwingHigh と SwingLow の間隔別（÷ATR・3分位。負＝高値ラインが安値ラインより下）")
        t["Swing間隔"] = tercile_label(t["SwingHigh-Low幅/ATR"], "ATR")
        rep.table(group_table(t, "Swing間隔"))

    rep.section("8. 決済理由別（SL/TP＝決済価格がその水準に到達。それ以外は「その他」）")
    reason = group_table(t, "決済理由")
    rep.table(reason)
    other = t[t["決済理由"] == "その他"]
    if len(other):
        rep.add(f"\n⚠️ SL/TP以外で決済されたトレードが {len(other)} 件あります"
                f"（exclusive_orders=False による反対側の逆指値の約定など）。先頭5件:")
        show = other[["EntryTime", "ExitTime", "方向", "EntryPrice", "ExitPrice", "SL", "TP", "損益pips"]].head(5).copy()
        for c in ["EntryPrice", "ExitPrice", "SL", "TP"]:
            show[c] = show[c].map(lambda v: f"{v:.5f}")   # 価格は小数5桁で表示（2桁だと区別できない）
        rep.table(show)
    else:
        rep.add("\n→ SL/TP以外で決済されたトレードはありません。")

    rep.section("9. 保有時間")
    hold = t.groupby(t["損益pips"].gt(0).map({True: "勝ち", False: "負け/ゼロ"}))["保有時間h"] \
            .agg(["count", "median", "mean", "max"])
    hold.columns = ["トレード数", "中央値[h]", "平均[h]", "最大[h]"]
    rep.table(hold)
    short = (t["保有時間h"] <= 1 / 60 + 1e-9).sum()
    rep.add(f"保有時間が1分以内のトレード: {short} 件"
            f"（同一バー内でSL/TP約定となり、結果があいまいになり得るトレードの目安）")

    rep.section("10. コスト感度（1往復あたりの追加コストを乗せたとき）")
    rows = {}
    for k in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        adj = p - k
        w, l = adj[adj > 0].sum(), adj[adj < 0].sum()
        rows[f"+{k:.1f} pips"] = {
            "合計pips": adj.sum(), "1トレード期待値pips": adj.mean(),
            "勝率[%]": (adj > 0).mean() * 100, "PF": w / abs(l) if l < 0 else np.nan,
            "t値": t_stat(adj),
        }
    rep.table(pd.DataFrame(rows).T)
    rep.add(f"→ 期待値が0になる追加コスト（損益分岐）: 約 {p.mean():.2f} pips / トレード"
            f"（逆指値エントリーのスリッページ等を見込むときの余裕）")

    if "MFE_ATR" in t.columns and t["MFE_ATR"].notna().any():
        rep.section("11. MAE/MFE（保有中の最大含み益 / 最大含み損。単位は ATR 倍）")
        rows = {}
        for name in ["TP", "SL", "その他"]:
            g = t[t["決済理由"] == name]
            if len(g) == 0:
                continue
            rows[f"{name}決済(n={len(g)})"] = {
                "MFE 中央値": g["MFE_ATR"].median(), "MFE 75%": g["MFE_ATR"].quantile(.75),
                "MAE 中央値": g["MAE_ATR"].median(), "MAE 75%": g["MAE_ATR"].quantile(.75),
            }
        rep.table(pd.DataFrame(rows).T)

        rep.add("\n▼ 最大含み益（MFE）が X ATR に到達した割合（現行のSLのもとでの実績）")
        levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        reach = pd.DataFrame({"全トレード[%]": [(t["MFE_ATR"] >= x).mean() * 100 for x in levels]},
                             index=[f"{x:.1f} ATR" for x in levels])
        sl_g = t[t["決済理由"] == "SL"]
        if len(sl_g):
            reach["SL決済のみ[%]"] = [(sl_g["MFE_ATR"] >= x).mean() * 100 for x in levels]
        rep.table(reach, float_fmt="{:,.1f}")
        if len(sl_g):
            rep.add("→ 「SL決済のみ」は、損切りになったトレードのうち一度は X ATR 有利に動いていた割合。"
                    "大きいほど、建値移動・トレーリングで救える余地があります。")

        tp_g = t[t["決済理由"] == "TP"]
        if len(tp_g):
            rep.add("\n▼ 利確したトレードが、利確までにSL距離の何割まで逆行したか（MAE ÷ SL距離）")
            ys = [0.25, 0.5, 0.75, 0.9]
            adverse = pd.DataFrame({"TP決済のうち到達した割合[%]": [(tp_g["MAE/SL距離"] >= y).mean() * 100 for y in ys]},
                                   index=[f"SL距離の {int(y * 100)}% 以上" for y in ys])
            rep.table(adverse, float_fmt="{:,.1f}")
            rep.add("→ 小さい割合なら、SLを今より狭くしても利確トレードへの影響が少ない可能性があります"
                    "（ただし負けトレードの損失は小さくなる一方、SLにかかる回数は増えます）。")
    return t


def main():
    ap = argparse.ArgumentParser(description="取引履歴CSVの診断")
    ap.add_argument("csv", nargs="?", default=None,
                    help="取引履歴CSV（ファイル名の一部でも可。省略すると最新の trade_history_*.csv）")
    ap.add_argument("--pip-size", type=float, default=0.0001, help="1pipの価格幅（EURUSD=0.0001 / USDJPY=0.01）")
    ap.add_argument("--price-tol", type=float, default=None, help="決済価格とSL/TPの一致判定の許容幅（既定: pip-size の1/10）")
    ap.add_argument("--out-dir", default="diagnosis", help="レポートと加工済みCSVの出力先フォルダ")
    ap.add_argument("--mae-mfe", action="store_true", help="MAE/MFE も計算する（元の1分足データが必要）")
    ap.add_argument("--data-path", default=None, help="1分足データのパス（既定: main_4H.py の Config.data_path）")
    args = ap.parse_args()

    tol = args.price_tol if args.price_tol is not None else args.pip_size / 10
    csv_path = resolve_csv(args.csv)
    t = load_trades(csv_path, args.pip_size, tol)

    if args.mae_mfe:
        t = add_mae_mfe(t, load_price_data(args.data_path))

    rep = Report()
    t = build_report(t, rep, csv_path.name)
    if not args.mae_mfe:
        rep.add("\n※ MAE/MFE の診断は --mae-mfe を付けて実行すると追加されます（元の1分足データが必要）。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnosis_report.txt").write_text("\n".join(rep.lines), encoding="utf-8-sig")
    t.to_csv(out_dir / "trades_enriched.csv", index=False, encoding="utf-8-sig")
    print(f"\n💾 レポート: {out_dir / 'diagnosis_report.txt'}\n💾 加工済み取引履歴: {out_dir / 'trades_enriched.csv'}")


if __name__ == "__main__":
    main()