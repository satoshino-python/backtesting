"""
FX版ミネルヴィニ戦略（通貨順位方式・ロング／ショート対称）の検証エンジン。
ルール全文は docs/MINERVINI_FX_RULES.md。日足で判定し、約定（ピボット・損切り・10日安値/高値のタッチ）は1分足で判定する。

流れ（すべて「その日の終値で確定した情報」だけを使う）
  1. 通貨強弱: 21ペア（検証できる15ペア＋USD経由の合成クロス6本）の63日対数リターンをボラ調整して通貨ごとに平均し、順位を付ける
  2. 候補ペア: ロング＝ベースが上位N・クォートが下位N、ショート＝その逆（Nは既定3）
  3. トレンドテンプレート（終値・50/150/200日線・200日線の傾き・52週レンジ内の位置）
  4. VCP: 日足のスイング（前後 swing_window 日のフラクタル）をジグザグにつなぎ、直近の調整幅（ATR単位）が毎回縮小していればピボットを確定
  5. 翌取引日の1分足でピボットにタッチしたらエントリー（窓開けは始値）。損切りは約定価格から ATR×2（固定）。
     損切りと「直近10日安値/高値」のうち先にタッチした方で手仕舞い（窓開けは始値）。同じ1分足でエントリーと損切りの両方に触れたら損切り優先
  6. ペア別の口座（10,000ドル）で1回のリスクを200ドル固定。損益はUSD、R＝損益÷200ドル。スプレッドとスワップを反映
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from minervini_cache import CACHE_DIR, FX_PAIRS, cache_path

CURRENCIES = ["USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CHF"]
# 検証できる15ペア（データのあるもの）。残り6ペアは強弱の計算だけに USD 経由の合成で使う（取引はしない）
SYNTHETIC = {  # 合成ペア: {USDペア: 係数} の対数価格の線形結合
    "CHFJPY": {"USDJPY": 1, "USDCHF": -1},
    "AUDCHF": {"AUDUSD": 1, "USDCHF": 1},
    "NZDCHF": {"NZDUSD": 1, "USDCHF": 1},
    "AUDNZD": {"AUDUSD": 1, "NZDUSD": -1},
    "EURNZD": {"EURUSD": 1, "NZDUSD": -1},
    "GBPNZD": {"GBPUSD": 1, "NZDUSD": -1},
}
# クォート通貨1単位のUSD価値を出す為替（"inv": 逆数）
QUOTE_USD = {"USD": None, "JPY": ("USDJPY", "inv"), "CHF": ("USDCHF", "inv"), "GBP": ("GBPUSD", "mul"), "AUD": ("AUDUSD", "mul")}


@dataclass(frozen=True)
class Params:
    start: str = "2021-01-01"            # 検証期間（取引日）。これより前は助走（200日線・52週・63日リターン用）
    end: str = "2025-12-31"
    ret_days: int = 63                   # 通貨強弱のリターン期間
    top_n: int = 3                       # 上位N買い・下位N売り
    min_gap: float = 0.0                 # ベースとクォートの強さ（S）の差の最低値
    ma_short: int = 50
    ma_mid: int = 150
    ma_long: int = 200
    ma_slope_days: int = 21              # 200日線が上昇/下降を続けている期間（約1か月＝21営業日）
    range_days: int = 252                # 52週
    long_pos: float = 0.70               # 52週レンジ内の位置（ロングは以上、ショートは 1-long_pos 以下）
    swing_window: int = 3                # スイング判定: 前後 N 日のフラクタル（確定は N 日後）
    min_contractions: int = 2            # 収縮の回数（調整幅が毎回縮小している連続回数）
    max_chain_days: int = 120            # 最初の収縮の起点から判定日まで
    max_pivot_age: int = 60              # ピボット（最後の収縮の起点）から判定日まで
    atr_period: int = 14
    sl_atr: float = 2.0
    exit_days: int | None = 10           # 直近N日安値/高値（None＝使わず2ATRの損切りだけ）
    cash: float = 10_000.0               # ペア別口座の初期資金
    risk_pct: float = 0.02
    max_ccy_positions: int | None = None  # 同じ通貨を同じ向きに持てる本数の上限（None＝制限なし）
    sides: tuple = ("long", "short")
    spread: bool = True                  # スプレッド（平均相対スプレッドを往復で1回分）を反映
    swap: bool = True                    # スワップを反映


# ---------------------------------------------------------------- データ
class PairData:
    """1分足（キャッシュ）と日足。期間は p.end までに切る。"""

    def __init__(self, pair, end):
        z = np.load(cache_path(pair))
        day = z["day"]
        end_i = np.datetime64(end, "D").astype("int64")
        keep = day <= end_i
        self.pair = pair
        self.ts, self.o, self.h, self.l, self.c = (z[k][keep] for k in ("ts", "o", "h", "l", "c"))
        if self.ts[0] < 10**17:                       # pandas 3 で作ったキャッシュは µs 単位。ns に揃える
            self.ts = self.ts * 1000
        day = day[keep]
        self.spread = float(z["spread"])
        brk = np.flatnonzero(np.diff(day)) + 1
        starts = np.concatenate([[0], brk])
        ends = np.concatenate([brk, [len(day)]])
        labels = day[starts]
        wk = (labels + 3) % 7 < 5                     # 1970-01-01 は木曜。月〜金の取引日だけ
        self.dstart, self.dend, labels = starts[wk], ends[wk], labels[wk]
        self.dates = pd.DatetimeIndex(labels.astype("datetime64[D]"))
        hi = np.maximum.reduceat(self.h, self.dstart)
        lo = np.minimum.reduceat(self.l, self.dstart)
        hi = np.where(self.dend - self.dstart > 0, hi, np.nan)
        self.daily = pd.DataFrame({"Open": self.o[self.dstart], "High": hi, "Low": lo, "Close": self.c[self.dend - 1]},
                                  index=self.dates)
        self.ts_ny = None

    def index_ny(self, idx):
        return pd.to_datetime(self.ts[idx], unit="ns", utc=True).tz_convert("America/New_York")


def load_all(end="2025-12-31", pairs=FX_PAIRS):
    return {p: PairData(p, end) for p in pairs}


# ---------------------------------------------------------------- 通貨強弱
def currency_strength(data, ret_days=63):
    """
    21ペアのボラ調整リターンから通貨の強さ S(i) を日次で返す（日付 × 通貨）。z = r ÷ (σ × √ret_days)。
    S(i) = 他の6通貨に対する z の平均（クォート側なら符号反転）。15ペアは実データ、6ペアは USD 経由の合成。
    """
    logc = pd.DataFrame({p: np.log(d.daily["Close"]) for p, d in data.items()})
    logc = logc.sort_index().ffill()
    for name, legs in SYNTHETIC.items():
        logc[name] = sum(k * logc[leg] for leg, k in legs.items())
    r = logc - logc.shift(ret_days)
    sigma = logc.diff().rolling(ret_days).std()
    z = r / (sigma * np.sqrt(ret_days))
    strength = pd.DataFrame(0.0, index=z.index, columns=CURRENCIES)
    count = dict.fromkeys(CURRENCIES, 0)
    for pair in z.columns:
        b, q = pair[:3], pair[3:]
        strength[b] += z[pair]
        strength[q] -= z[pair]
        count[b] += 1
        count[q] += 1
    for c in CURRENCIES:
        strength[c] /= count[c]
    strength[z.isna().any(axis=1)] = np.nan
    return strength


# ---------------------------------------------------------------- 日足の指標
def wilder_atr(h, l, c, period):
    prev = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    tr[0] = h[0] - l[0]
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False, min_periods=period).mean().to_numpy()


def vcp_pivots(h, l, atr, window, n_contr, max_chain, max_age):
    """
    各日の終値時点で見えている VCP のピボットを返す（pivot_long, pivot_short。無い日は NaN）。
    スイング（前後 window 日のフラクタル）は window 日後に確定。ジグザグ（高値・安値が交互）の末尾から n_contr 回分の
    調整（高値→安値の幅。戻りは安値→高値）をATR単位（安値側/高値側の確定時点のATR）で測り、毎回縮小していればピボットが決まる。
      ロング: 末尾が安値。ピボット＝最後の調整の起点の高値。その高値をまだ超えていないこと
      ショート: 末尾が高値。ピボット＝最後の戻りの起点の安値。その安値をまだ割っていないこと
    """
    n = len(h)
    pl = np.full(n, np.nan)
    ps = np.full(n, np.nan)
    zig = []   # [種別(+1高値/-1安値), 日の位置, 価格]

    def add(typ, idx, price):
        if zig and zig[-1][0] == typ:
            if (typ == 1 and price > zig[-1][2]) or (typ == -1 and price < zig[-1][2]):
                zig[-1] = [typ, idx, price]
        else:
            zig.append([typ, idx, price])

    w = window
    for t in range(n):
        j = t - w
        if j >= w:
            if h[j] > h[j - w:j].max() and h[j] >= h[j + 1:j + w + 1].max():
                add(1, j, h[j])
            if l[j] < l[j - w:j].min() and l[j] <= l[j + 1:j + w + 1].min():
                add(-1, j, l[j])
        if len(zig) < 2 * n_contr:
            continue
        end = zig[-1]
        pairs = [(zig[-2 * (m + 1)], zig[-2 * m - 1]) for m in range(n_contr)]   # 新しい順の (起点, 終点)
        if t - pairs[-1][0][1] > max_chain or t - pairs[0][0][1] > max_age:
            continue
        depth = []
        ok = True
        for a, b in pairs:
            a_ = atr[b[1]]
            d = abs(a[2] - b[2])
            if not (a_ > 0) or d <= 0:
                ok = False
                break
            depth.append(d / a_)
        if not ok or any(depth[m] >= depth[m + 1] for m in range(n_contr - 1)):
            continue
        start = pairs[0][0]
        if end[0] == -1 and start[0] == 1:
            if h[start[1] + 1:t + 1].max() < start[2]:
                pl[t] = start[2]
        elif end[0] == 1 and start[0] == -1:
            if l[start[1] + 1:t + 1].min() > start[2]:
                ps[t] = start[2]
    return pl, ps


def daily_signals(pd_, rank_ok, p):
    """
    ペアの日足ごとのシグナル（判定日の終値で確定→翌取引日に有効）。
    返す DataFrame: Side（1ロング/-1ショート/0なし）, Pivot, Atr, LL, HH（直近N日安値/高値＝前日まで）
    """
    d = pd_.daily
    c, h, l = d["Close"].to_numpy(), d["High"].to_numpy(), d["Low"].to_numpy()
    cs = d["Close"]
    s50, s150, s200 = (cs.rolling(m).mean() for m in (p.ma_short, p.ma_mid, p.ma_long))
    hi52 = d["High"].rolling(p.range_days).max()
    lo52 = d["Low"].rolling(p.range_days).min()
    pos = (cs - lo52) / (hi52 - lo52)
    atr = wilder_atr(h, l, c, p.atr_period)
    slope_up = s200 > s200.shift(p.ma_slope_days)
    slope_dn = s200 < s200.shift(p.ma_slope_days)
    tl = ((cs > s150) & (cs > s200) & (s150 > s200) & slope_up & (s50 > s150) & (s50 > s200) & (cs > s50)
          & (pos >= p.long_pos))
    ts_ = ((cs < s150) & (cs < s200) & (s150 < s200) & slope_dn & (s50 < s150) & (s50 < s200) & (cs < s50)
           & (pos <= 1 - p.long_pos))
    pl, ps = vcp_pivots(h, l, atr, p.swing_window, p.min_contractions, p.max_chain_days, p.max_pivot_age)
    long_ok = tl.to_numpy() & rank_ok[0] & ~np.isnan(pl) & ("long" in p.sides)
    short_ok = ts_.to_numpy() & rank_ok[1] & ~np.isnan(ps) & ("short" in p.sides)
    side = np.where(long_ok, 1, np.where(short_ok, -1, 0))
    pivot = np.where(side == 1, pl, np.where(side == -1, ps, np.nan))
    if p.exit_days:
        ll = d["Low"].shift(1).rolling(p.exit_days).min().to_numpy()
        hh = d["High"].shift(1).rolling(p.exit_days).max().to_numpy()
    else:
        ll = hh = np.full(len(d), np.nan)
    return pd.DataFrame({"Side": side, "Pivot": pivot, "Atr": atr, "LL": ll, "HH": hh, "Template_L": tl.to_numpy(),
                         "Template_S": ts_.to_numpy(), "PivotL": pl, "PivotS": ps}, index=d.index)


def rank_flags(strength, pair, dates, p):
    """ペアの各日について (ロング候補, ショート候補) の bool 配列。強弱は判定日の終値時点のもの"""
    b, q = pair[:3], pair[3:]
    rk = strength.rank(axis=1, ascending=False)
    top = rk <= p.top_n
    bot = rk >= len(CURRENCIES) + 1 - p.top_n
    gap = strength[b] - strength[q]
    ok_l = (top[b] & bot[q] & (gap >= p.min_gap)).reindex(dates, method="ffill").fillna(False).to_numpy(bool)
    ok_s = (bot[b] & top[q] & (-gap >= p.min_gap)).reindex(dates, method="ffill").fillna(False).to_numpy(bool)
    return ok_l, ok_s


# ---------------------------------------------------------------- 約定（1分足）
def find_fill(pd_, d, side, pivot):
    """取引日 d の1分足で、ピボットに最初にタッチした分の (位置, 約定価格)。窓開けは始値。なければ None"""
    a, b = pd_.dstart[d], pd_.dend[d]
    if side == 1:
        hit = np.flatnonzero(pd_.h[a:b] >= pivot)
        if not len(hit):
            return None
        i = a + hit[0]
        return i, max(pivot, pd_.o[i])
    hit = np.flatnonzero(pd_.l[a:b] <= pivot)
    if not len(hit):
        return None
    i = a + hit[0]
    return i, min(pivot, pd_.o[i])


def simulate_exit(pd_, sig, d0, ie, entry, side, sl):
    """
    エントリー後の手仕舞い。毎日、損切りと前日までの10日安値/高値のうち近い方（先にタッチする方）を逆指値として扱う。
    エントリーした1分足自体で逆指値に触れていたら損切り優先で手仕舞い。窓開けは始値。
    戻り値: (分の位置, 価格, 理由, 取引日の位置)。期間終了まで持ち越したら最終1分足の終値（理由「期間終了」）。
    """
    nd = len(pd_.dstart)
    ll, hh = sig["LL"].to_numpy(), sig["HH"].to_numpy()
    for d in range(d0, nd):
        a, b = pd_.dstart[d], pd_.dend[d]
        if d == d0:
            a = ie                                     # エントリーした分も含める（保守的）
        if side == 1:
            trail = ll[d]
            stop = max(sl, trail) if trail == trail else sl
            hit = np.flatnonzero(pd_.l[a:b] <= stop)
        else:
            trail = hh[d]
            stop = min(sl, trail) if trail == trail else sl
            hit = np.flatnonzero(pd_.h[a:b] >= stop)
        if len(hit):
            i = a + hit[0]
            if side == 1:
                px = min(stop, pd_.o[i]) if i > ie else stop
                reason = "損切り" if sl >= (trail if trail == trail else -np.inf) else "10日安値"
            else:
                px = max(stop, pd_.o[i]) if i > ie else stop
                reason = "損切り" if sl <= (trail if trail == trail else np.inf) else "10日高値"
            return i, px, reason, d
    last = len(pd_.c) - 1
    return last, pd_.c[last], "期間終了", nd - 1


def quote_usd_at(data, quote, ts):
    """クォート通貨1単位のUSD価値（ts 時点の直近の1分足終値）"""
    spec = QUOTE_USD[quote]
    if spec is None:
        return 1.0
    pair, how = spec
    cd = data[pair]
    i = max(np.searchsorted(cd.ts, ts, side="right") - 1, 0)
    v = cd.c[i]
    return 1.0 / v if how == "inv" else v


# ---------------------------------------------------------------- 全体のシミュレーション
def prepare(data, p):
    """通貨強弱と、ペアごとの日足シグナルを作る"""
    strength = currency_strength(data, p.ret_days)
    sigs = {}
    for pair, pdat in data.items():
        flags = rank_flags(strength, pair, pdat.daily.index, p)
        sigs[pair] = daily_signals(pdat, flags, p)
    return strength, sigs


def run(data, p=Params(), prepared=None, swap_tables=None):
    """
    検証を実行して取引履歴（DataFrame）を返す。
    ペアごとの口座は独立。同じペアは1ポジションまで。同じ日に複数ペアが約定する場合は約定時刻順に処理し、
    max_ccy_positions があれば「同じ通貨を同じ向きに持つ本数」がその数に達していたら見送る。
    """
    strength, sigs = prepared if prepared is not None else prepare(data, p)
    start = np.datetime64(p.start, "D")
    end = np.datetime64(p.end, "D")
    events = []   # (注文日, ペア, 判定日の位置)
    for pair, sig in sigs.items():
        dates = data[pair].daily.index.to_numpy().astype("datetime64[D]")
        idx = np.flatnonzero(sig["Side"].to_numpy() != 0)
        for k in idx:
            if k + 1 < len(dates) and start <= dates[k + 1] <= end:
                events.append((dates[k + 1], pair, k))
    events.sort(key=lambda e: (e[0], e[1]))

    busy_until = {pair: -1 for pair in data}            # ペアごとの直近の手仕舞い時刻（UTC ns）
    open_pos = []                                        # (手仕舞い時刻, ベース通貨, クォート通貨, 向き)
    trades = []
    i = 0
    while i < len(events):
        day = events[i][0]
        j = i
        while j < len(events) and events[j][0] == day:
            j += 1
        fills = []
        for _, pair, k in events[i:j]:
            pdat, sig = data[pair], sigs[pair]
            d = k + 1
            if busy_until[pair] > pdat.ts[pdat.dstart[d]]:
                continue                                 # 判定日の終値時点でまだポジションを持っている
            side = int(sig["Side"].iat[k])
            f = find_fill(pdat, d, side, sig["Pivot"].iat[k])
            if f is not None:
                fills.append((pdat.ts[f[0]], pair, k, f[0], f[1]))
        fills.sort()
        for ts_e, pair, k, ie, entry in fills:
            pdat, sig = data[pair], sigs[pair]
            side = int(sig["Side"].iat[k])
            base, quote = pair[:3], pair[3:]
            if p.max_ccy_positions is not None:
                open_pos = [x for x in open_pos if x[0] > ts_e]
                buy, sell = (base, quote) if side == 1 else (quote, base)
                nb = sum(1 for x in open_pos if (x[1] if x[3] == 1 else x[2]) == buy)
                ns = sum(1 for x in open_pos if (x[2] if x[3] == 1 else x[1]) == sell)
                if nb >= p.max_ccy_positions or ns >= p.max_ccy_positions:
                    continue
            atr = sig["Atr"].iat[k]
            sl = entry - side * p.sl_atr * atr
            ix, px, reason, dx = simulate_exit(pdat, sig, k + 1, ie, entry, side, sl)
            busy_until[pair] = pdat.ts[ix]
            open_pos.append((pdat.ts[ix], base, quote, side))
            risk = p.cash * p.risk_pct
            q_in = quote_usd_at(data, quote, ts_e)
            q_out = quote_usd_at(data, quote, pdat.ts[ix])
            units = risk / (p.sl_atr * atr * q_in)
            cost = pdat.spread * (entry + px) / 2 if p.spread else 0.0     # 往復でスプレッド1回分（片道ずつ半分）
            gross_q = units * (side * (px - entry))
            pnl = (gross_q - units * cost) * q_out
            trades.append({
                "Pair": pair, "Side": "Long" if side == 1 else "Short", "Size": side * units,
                "EntryTime": ts_e, "ExitTime": pdat.ts[ix], "EntryPrice": entry, "ExitPrice": px,
                "SL": sl, "Pivot": sig["Pivot"].iat[k], "ATR": atr, "ExitReason": reason,
                "GapATR": side * (entry - sig["Pivot"].iat[k]) / atr,
                "TrailATR": side * (entry - (sig["LL"] if side == 1 else sig["HH"]).iat[k + 1]) / atr,   # 10日安値/高値までの距離（ATR）
                "SignalDay": pdat.daily.index[k], "ExitDay": pdat.daily.index[dx],
                "SpreadUSD": units * cost * q_out, "PnL": pnl,
                "StrengthBase": strength[base].reindex([pdat.daily.index[k]], method="ffill").iat[0],
                "StrengthQuote": strength[quote].reindex([pdat.daily.index[k]], method="ffill").iat[0],
            })
        i = j
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    for col in ("EntryTime", "ExitTime"):
        df[col] = pd.to_datetime(df[col], unit="ns", utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values(["ExitTime", "EntryTime"]).reset_index(drop=True)
    if p.swap:
        df = apply_swap(df, data, swap_tables)
    else:
        df["PnL_Gross"], df["Interest"] = df["PnL"], 0.0
    df["R"] = df["PnL"] / (p.cash * p.risk_pct)
    df["R_Gross"] = df["PnL_Gross"] / (p.cash * p.risk_pct)
    df["Year"] = df["ExitTime"].dt.year
    return df


def apply_swap(trades, data, tables=None):
    """スワップ（swap_costs の FX 表）を取引ごとに反映する。PnL_Gross（反映前）・InterestJPY・Interest（USD）を足す"""
    from swap_costs import SWAP_SOURCES, load_interest_table, apply_interest
    usdjpy = pd.Series(data["USDJPY"].c, index=data["USDJPY"].index_ny(slice(None)))
    out = []
    for pair, g in trades.groupby("Pair", sort=False):
        csv, lot, has_pair = SWAP_SOURCES[pair]
        table = (tables or {}).get(pair) or load_interest_table(csv, pair if has_pair else None)
        out.append(apply_interest(g, table, usdjpy, lot_units=lot))
    res = pd.concat(out).sort_values(["ExitTime", "EntryTime"]).reset_index(drop=True)
    return res


# ---------------------------------------------------------------- 集計
def summarize(r, years=None):
    r = np.asarray(r, dtype=float)
    if not len(r):
        return {"トレード数": 0, "合計R": 0.0, "PF": np.nan, "勝率[%]": np.nan, "平均R": np.nan, "最大連敗": 0}
    wins, losses = r[r > 0], r[r < 0]
    streak = best = 0
    for v in r:
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    return {"トレード数": len(r), "合計R": r.sum(), "PF": wins.sum() / abs(losses.sum()) if len(losses) else np.nan,
            "勝率[%]": len(wins) / len(r) * 100, "平均R": r.mean(), "最大連敗": best}


def max_dd(r):
    """決済順のRの累積曲線の最大ドローダウン（R）"""
    if not len(r):
        return 0.0
    cum = np.cumsum(np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float((peak - cum).max())


def table(trades, by, col="R"):
    rows = {}
    for key, g in trades.groupby(by):
        s = summarize(g[col].to_numpy())
        s["最大DD(R)"] = max_dd(g[col].to_numpy())
        rows[key] = s
    return pd.DataFrame(rows).T


def overall(trades, col="R"):
    s = summarize(trades[col].to_numpy())
    s["最大DD(R)"] = max_dd(trades[col].to_numpy())
    return s


def concurrency(trades):
    """同時に持っていたポジション数の最大と、同じ通貨を同じ向きに持った本数の最大"""
    ev = []
    for t in trades.itertuples():
        base, quote = t.Pair[:3], t.Pair[3:]
        buy, sell = (base, quote) if t.Side == "Long" else (quote, base)
        ev.append((t.EntryTime, 1, buy, sell))
        ev.append((t.ExitTime, -1, buy, sell))
    ev.sort(key=lambda e: (e[0], e[1]))
    n = 0
    cb, cs = {}, {}
    mx = mb = 0
    for _, k, buy, sell in ev:
        n += k
        cb[buy] = cb.get(buy, 0) + k
        cs[sell] = cs.get(sell, 0) + k
        mx = max(mx, n)
        mb = max(mb, cb[buy], cs[sell])
    return {"最大同時保有": mx, "同通貨同方向の最大": mb}
