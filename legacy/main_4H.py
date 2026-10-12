import os
import re
import zipfile
import io
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import bokeh
from backtesting import Backtest, Strategy
from html import escape
import webbrowser
import time
import warnings


# ===== 変更するパラメータはここだけ =====
# 個別に上書きする場合の例:
#   CFG = Config(window=7, data_path=Path("histData/EURUSD"))
#   CFG = Config(start_date="2026-08-10", end_date="2026-08-20")  # この期間だけ検証
@dataclass(frozen=True)
class Config:
    data_path: Path = Path("histData/EURUSD")
    price_side: str = "mid"          # "mid" / "bid" / "ask"
    session_start_hour: int = 17     # NYクローズ基準（判定足の区切りの起点）
    signal_hours: int = 4            # 判定足の長さ（時間）。24の約数（1,2,3,4,6,8,12,24）のみ可
    cash: float = 10_000
    size: int = 10000                 # 1回のトレードのロット数
    window: int = 18                 # Swing Point の前後判定期間（判定足の本数。4時間足18本 ≒ 3日）
    atr_period: int = 18             # ATR 期間（判定足の本数）
    sl_atr_multiplier: float = 1.0
    tp_atr_multiplier: float = 1.5
    price_decimals: int = 5
    csv_filename: str = "trade_history.csv"
    html_filename: str = "swing_breakout_chart.html"
    # 検証期間の指定（"YYYY-MM-DD" 文字列）。読み込んだデータのうち、この期間の
    # 取引日（NY 17:00 区切り）だけに絞ってバックテストする。片方だけの指定や、
    # 両方 None（絞り込みなし＝読み込んだ全期間を検証）も可能。
    # 例: start_date="2026-08-10", end_date="2026-08-20" → その11日間だけ検証
    # 注意: window（前後N本）の判定のため、指定期間の前後 window 本分（判定足）は
    #       シグナルが出るまでの助走期間として使われ、その分トレード数は
    #       少なくなる点に留意（元データ自体は指定期間より広く読み込んでおく必要あり）
    start_date: str | None = None
    end_date: str | None = None
    # "1min": SwingHigh/Low・ATRの判定は判定足（signal_hours 時間足）ベースのまま、
    #         実際のエントリー・利確・損切りの執行は1分足のOHLCで行う（より現実的・推奨）
    # "signal": 判定・執行とも判定足ベース
    execution_timeframe: str = "1min"
    # チャート（HTML）の表示単位。バックテストの計算精度には影響しない（表示だけの設定）。
    # "signal": 1分足で執行していても、チャートは判定足（4時間足）のローソク足にまとめて表示する
    #           （時刻表示は実時刻から (session_start_hour % signal_hours) 時間ずれた表示専用の値）
    # "native": execution_timeframe と同じ単位でそのまま表示（1分足実行時はローソク足が
    #           多くなるため、20,000本を超える場合は自動的に間引き表示になる）
    chart_timeframe: str = "signal"
    # 判定足実行モード（execution_timeframe="signal"）時の Swing Point の表示形式。
    # "dash": 短い水平線（既定・通貨の価格水準に依存せず縦軸が潰れない）
    # "circle": 丸マーカー（従来表示。EURUSDなど価格が1前後の通貨では縦軸が広がる）
    swing_point_style: str = "dash"
    #レバレッジ設定
    margin: float = 1/25
    # SL×TP 組み合わせ表（グリッド検証）。True にすると、通常の単発検証（チャート・CSV出力）の
    # 代わりに、下の SL倍率 × TP倍率 の組み合わせ（利確幅 > 損切り幅 のものだけ）を検証して、
    # 主要指標の一覧表を出力する。
    # 1回の検証に時間がかかる（特に1分足執行で期間が長い場合）ため、まず短い期間や
    # 少ない組み合わせで所要時間を確認してから広げること。
    grid_enabled: bool = False
    grid_sl_values: tuple = (0.8, 1.0, 1.2, 1.5)   # 損切り幅 = ATR × この値 の候補
    grid_tp_values: tuple = (1.0, 1.5, 2.0, 2.5)   # 利確幅 = ATR × この値 の候補
    grid_csv_filename: str = "sl_tp_grid.csv"
    # True にすると、組み合わせごとの取引履歴（単発検証の trade_history.csv と同じ形式）を
    # grid_trades_dir フォルダに「trades_SL1.2_TP2.0.csv」のような名前で保存する。
    # （トレードが0件の組み合わせは保存しない）
    grid_save_trades: bool = True
    grid_trades_dir: str = "grid_trades"


CFG = Config(
    start_date="2021-01-01", 
    end_date="2025-12-31",
    sl_atr_multiplier=1.5,
    tp_atr_multiplier=2.5,
    # True : SL×TP の組み合わせ表を出力（チャートは作らない。sl/tp_atr_multiplier は使われない）
    # False: 上の sl/tp_atr_multiplier で単発検証（チャート・取引履歴CSVを出力）
    grid_enabled=False,
    )  

# Bokeh 3.x 系の互換性警告チェック
if bokeh.__version__.startswith('3.'):
    print("=" * 60)
    print(f"⚠️  検出された Bokeh バージョン: {bokeh.__version__}")
    print("⚠️  backtesting.py は Bokeh 3.x で取引マーク（▲/▼矢印）が非表示になる問題があります。")
    print("👉 ターミナルで以下のコマンドを実行して Bokeh 2.4.3 をインストールしてください:")
    print("   pip install \"bokeh==2.4.3\"")
    print("=" * 60 + "\n")


# GMOクリック証券のFXヒストリカルデータ（1分足）のCSV列名
# 日時, 始値(BID), 高値(BID), 安値(BID), 終値(BID), 始値(ASK), 高値(ASK), 安値(ASK), 終値(ASK)
_GMO_ENCODING = "cp932"
_GMO_COLUMNS = [
    "Datetime",
    "Open_BID", "High_BID", "Low_BID", "Close_BID",
    "Open_ASK", "High_ASK", "Low_ASK", "Close_ASK",
]


def _read_single_gmo_csv(file_like_or_path):
    """
    GMOクリック証券形式の1日分CSV（Shift-JIS/cp932）を1つ読み込む。
    ファイルパス、または bytes を渡せる file-like オブジェクトのどちらでも可。
    """
    df = pd.read_csv(
        file_like_or_path,
        encoding=_GMO_ENCODING,
        header=0,
        names=_GMO_COLUMNS,
        skiprows=1,  # ヘッダー行（日本語）を読み飛ばす
    )
    return df


def load_gmo_click_1min_data(
    source_path,
    price_side="mid",
    source_tz="Asia/Tokyo",
    target_tz="America/New_York",
    price_decimals=5,
):
    """
    GMOクリック証券の1分足ヒストリカルデータ（ZIPまたはフォルダ）を読み込み、
    backtesting.py が要求するデータフレーム形式（Open/High/Low/Close/Volume,
    DatetimeIndex）に変換する。

    GMOクリック証券のタイムスタンプは日本時間（source_tz）で記録されているため、
    ここで NY時間（target_tz）に変換する（サマータイムも自動考慮）。

    Parameters
    ----------
    source_path : str or Path
        - .zip ファイルのパス（例: histData/USDJPY_202608.zip）
          → 中身の *.csv を解凍せずにそのまま全て読み込み、日付順に結合する
        - フォルダのパス
          → フォルダ内（サブフォルダ含む）に .zip ファイルが1つ以上あれば、
            それら全ての ZIP の中身の *.csv を全て解凍せずに読み込み、
            日付順（ZIPファイル名→CSVファイル名の順）に結合する
            （例: histData/ に USDJPY_202608.zip, USDJPY_202609.zip ... を
            まとめて置いておけば、複数ヶ月分を自動的に結合できる）
          → フォルダ内に ZIP が無い場合は、フォルダ内（サブフォルダ含む）の
            *.csv を直接読み込み、日付順に結合する（解凍済みデータ用）
    price_side : str
        "mid"（既定・仲値 = (BID+ASK)/2） / "bid" / "ask" のいずれかを OHLC として採用する。
        スプレッドコストは別途 avg_relative_spread として返すので、
        "mid" を使い、それを commission としてバックテストに反映するのが基本の使い方。

    Returns
    -------
    (df, avg_relative_spread) のタプル
        df : DatetimeIndex（NY時間）を持つ Open/High/Low/Close/Volume の DataFrame
        avg_relative_spread : ASKとBIDの差を仲値に対する比率で表した平均値
            （例: 0.00002 なら仲値の 0.002%）。
            commission = avg_relative_spread / 2 として Backtest に渡すと、
            「仲値ベースで売買しつつ、往復のスプレッドコストをコミッションとして
            エントリー・イグジットの両方に計上する」近似になる
            （backtesting.py の commission は往復で2回課金されるため 1/2 する）。
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"❌ 指定されたパスが見つかりません: {source_path.resolve()}")

    print(f"📂 データ読み込み中: {source_path.resolve()}")

    daily_frames = []

    def _read_csvs_from_zip(zip_path):
        """1つのZIPファイルの中身のCSVを全て読み込んで daily_frames に積む"""
        with zipfile.ZipFile(zip_path, "r") as zf:
            # ファイル名（例: USDJPY_20260803.csv）でソートして日付順を担保
            csv_names = sorted(
                n for n in zf.namelist()
                if n.lower().endswith(".csv") and not os.path.basename(n).startswith(".")
            )
            if not csv_names:
                raise ValueError(f"❌ ZIP内にCSVファイルが見つかりません: {zip_path}")
            for name in csv_names:
                with zf.open(name) as f:
                    daily_frames.append(_read_single_gmo_csv(io.BytesIO(f.read())))
            return len(csv_names)

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        # 単一のZIPファイルを指定された場合
        n_csv = _read_csvs_from_zip(source_path)
        print(f"ℹ️ ZIP「{source_path.name}」内の {n_csv} 個のCSVファイルを読み込みました。")

    elif source_path.is_dir():
        # フォルダ内の ZIP ファイルを（サブフォルダ含めて）全て探す。
        # 複数ヶ月分の月次ZIPをまとめて置いておくケースを想定し、
        # ファイル名順（＝時系列順になる想定）に処理する。
        zip_files = sorted(source_path.rglob("*.zip"))

        if zip_files:
            total_csv = 0
            for zp in zip_files:
                n_csv = _read_csvs_from_zip(zp)
                total_csv += n_csv
                print(f"ℹ️ ZIP「{zp.name}」内の {n_csv} 個のCSVファイルを読み込みました。")
            print(f"ℹ️ 合計 {len(zip_files)} 個のZIP、{total_csv} 個のCSVファイルを結合しました。")
        else:
            # ZIPが無ければ、既に解凍済みのCSVを直接探す
            csv_files = sorted(source_path.rglob("*.csv"))
            if not csv_files:
                raise ValueError(f"❌ フォルダ内にZIPまたはCSVファイルが見つかりません: {source_path}")
            for f in csv_files:
                daily_frames.append(_read_single_gmo_csv(f))
            print(f"ℹ️ フォルダ内の {len(csv_files)} 個のCSVファイルを読み込みました。")
    else:
        raise ValueError("❌ source_path は .zip ファイルまたはフォルダを指定してください。")

    df = pd.concat(daily_frames, ignore_index=True)

    # 日時変換・重複排除・並び替え
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.drop_duplicates(subset="Datetime").sort_values("Datetime").set_index("Datetime")

    # 日本時間 → NY時間 へタイムゾーン変換（サマータイム自動考慮）
    df.index = df.index.tz_localize(source_tz).tz_convert(target_tz)

    # 平均相対スプレッド（終値ベース）を算出。commission算出に利用する。
    mid_close_for_spread = (df["Close_BID"] + df["Close_ASK"]) / 2
    relative_spread = (df["Close_ASK"] - df["Close_BID"]) / mid_close_for_spread
    avg_relative_spread = float(relative_spread.mean())

    price_side = price_side.lower()
    if price_side == "bid":
        o, h, l, c = df["Open_BID"], df["High_BID"], df["Low_BID"], df["Close_BID"]
    elif price_side == "ask":
        o, h, l, c = df["Open_ASK"], df["High_ASK"], df["Low_ASK"], df["Close_ASK"]
    elif price_side == "mid":
        o = (df["Open_BID"] + df["Open_ASK"]) / 2
        h = (df["High_BID"] + df["High_ASK"]) / 2
        l = (df["Low_BID"] + df["Low_ASK"]) / 2
        c = (df["Close_BID"] + df["Close_ASK"]) / 2
    else:
        raise ValueError("❌ price_side は 'bid' / 'ask' / 'mid' のいずれかを指定してください。")

    out = pd.DataFrame({
        "Open": o.astype(float).round(price_decimals),
        "High": h.astype(float).round(price_decimals),
        "Low": l.astype(float).round(price_decimals),
        "Close": c.astype(float).round(price_decimals),
    })
    # GMOクリック証券の1分足データには出来高(Volume)が含まれないため 0 で埋める
    out["Volume"] = 0

    out = out.dropna()

    print(
        f"✅ データ読み込み完了（{price_side.upper()}, {target_tz}）: "
        f"{out.index.min()} ～ {out.index.max()} ({len(out):,} 行)"
    )
    print(f"ℹ️ 平均相対スプレッド: {avg_relative_spread * 100:.5f}% (commission換算: {avg_relative_spread / 2 * 100:.5f}%)")

    return out, avg_relative_spread


def get_trading_day_label(index_ny, session_start_hour=17):
    """
    NY時間のタイムゾーン付き DatetimeIndex から、各バーが属する「取引日」の
    日付ラベル（タイムゾーンなし、月〜金の通常の曜日になる側）を算出する。
    検証期間（start_date / end_date）の絞り込みなど、取引日ラベルが必要な箇所で使う共通関数。
    """
    shifted_index = index_ny - pd.Timedelta(hours=session_start_hour)
    trading_day = pd.to_datetime(shifted_index.date) + pd.Timedelta(days=1)
    return trading_day


def get_signal_bar_label(index_ny, session_start_hour=17, signal_hours=4):
    """
    NY時間のタイムゾーン付き DatetimeIndex から、各バーが属する「判定足」の
    開始時刻ラベル（タイムゾーンなし・NY現地時刻）を返す。

    判定足は NY session_start_hour 時（既定 17:00＝NYクローズ）を起点に signal_hours 時間ごとに
    区切る。4時間足なら 17-21時 / 21-1時 / 1-5時 / 5-9時 / 9-13時 / 13-17時 の6本/日。

    現地時刻（壁時計）で計算するため、サマータイム切替の影響を受けない。
    resample_to_signal_bars() と map_signals_to_1min() で同一のロジックを使うための
    共通関数（ここがズレると判定足シグナルと1分足の対応がズレるため重要）。
    """
    wall = index_ny.tz_localize(None)
    offset = pd.Timedelta(hours=session_start_hour)
    return ((wall - offset).floor(pd.Timedelta(hours=signal_hours)) + offset).rename(None)


def signal_bar_trading_day(signal_index, session_start_hour=17):
    """
    判定足の開始時刻ラベル（タイムゾーンなし）から、その足が属する「取引日」の
    日付ラベルを返す。get_trading_day_label() と同じ規則
    （NY 17:00〜翌17:00 を、大半が含まれる側の日付＝月〜金で表す）で、
    検証期間（start_date / end_date）の絞り込みに使う。
    """
    shifted = signal_index - pd.Timedelta(hours=session_start_hour)
    return shifted.normalize() + pd.Timedelta(days=1)


def make_chart_index_for_signal_bars(index_ny, session_start_hour=17, signal_hours=4):
    """
    【チャート表示専用】1分足のタイムスタンプを、bt.plot(resample=f"{signal_hours}h") の
    区切りが判定足の区切りと一致するように並べ替えた、タイムゾーンなしの
    DatetimeIndex を返す。

    bt.plot(resample=...) は「0:00起点」で signal_hours 時間ごとに集約するため、
    NY session_start_hour 時起点の判定足（4時間足なら 17,21,1,5,9,13時 区切り）とは
    境界がズレる。そこで NY現地時刻を (session_start_hour % signal_hours) 時間だけ戻した
    仮の時刻を使い、集約の境界を判定足の境界と一致させる
    （4時間足・17時起点なら 1時間戻す）。並び順・一意性は保たれる。
    表示専用なので約定計算や集計結果には使わない。
    """
    shift = pd.Timedelta(hours=session_start_hour % signal_hours)
    return pd.DatetimeIndex(index_ny.tz_localize(None) - shift, name=index_ny.name)


def resample_to_signal_bars(df_1min, session_start_hour=17, signal_hours=4):
    """
    NY時間のTZ付き1分足データを判定足（既定は4時間足）にリサンプルする。

    FXの慣習に合わせ、区切りの起点は NY時間 17:00（NYクローズ）とする
    （session_start_hour=17 が既定値）。4時間足なら 17-21時 / 21-1時 / 1-5時 /
    5-9時 / 9-13時 / 13-17時 の6本/日になる。

    ラベルは各足の「開始時刻」（タイムゾーンなし・NY現地時刻）。

    Returns
    -------
    Open/High/Low/Close/Volume を持つ DataFrame
    """
    bar_label = get_signal_bar_label(df_1min.index, session_start_hour, signal_hours)

    bars = df_1min.groupby(bar_label).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    bars.index.name = "Datetime"

    print(f"✅ {signal_hours}時間足へのリサンプル完了（区切りの起点: NY時間 {session_start_hour:02d}:00）: "
          f"{bars.index.min().strftime('%Y-%m-%d %H:%M')} ～ {bars.index.max().strftime('%Y-%m-%d %H:%M')} "
          f"({len(bars):,} 本)")
    return bars


def get_swing_high_marks(high_series, window=5):
    """
    前後 N 本の最高値となるポイント（Swing High）を抽出
    """
    highs = pd.Series(high_series).values
    n = len(highs)
    marks = np.full(n, np.nan)
    
    for i in range(window, n - window):
        sub = highs[i - window : i + window + 1]
        if highs[i] == np.max(sub):
            marks[i] = highs[i]
            
    return marks


def get_swing_low_marks(low_series, window=5):
    """
    前後 N 本の最安値となるポイント（Swing Low）を抽出
    """
    lows = pd.Series(low_series).values
    n = len(lows)
    marks = np.full(n, np.nan)
    
    for i in range(window, n - window):
        sub = lows[i - window : i + window + 1]
        if lows[i] == np.min(sub):
            marks[i] = lows[i]
            
    return marks


def get_latest_swing_high(high_series, window=5):
    """
    リアルタイムで利用可能な最新の確定 Swing High 価格ライン
    （ブレイクアウト時にも値が消失しないよう最新のSwing Highを保持します）
    """
    highs = pd.Series(high_series).values
    n = len(highs)
    marks = get_swing_high_marks(high_series, window)
    
    latest_sh = np.full(n, np.nan)
    current_val = np.nan
    
    for t in range(n):
        check_idx = t - window
        if check_idx >= 0 and not np.isnan(marks[check_idx]):
            current_val = marks[check_idx]
        latest_sh[t] = current_val
        
    return latest_sh


def get_latest_swing_low(low_series, window=5):
    """
    リアルタイムで利用可能な最新の確定 Swing Low 価格ライン
    （ブレイクアウト時にも値が消失しないよう最新のSwing Lowを保持します）
    """
    lows = pd.Series(low_series).values
    n = len(lows)
    marks = get_swing_low_marks(low_series, window)
    
    latest_sl = np.full(n, np.nan)
    current_val = np.nan
    
    for t in range(n):
        check_idx = t - window
        if check_idx >= 0 and not np.isnan(marks[check_idx]):
            current_val = marks[check_idx]
        latest_sl[t] = current_val
        
    return latest_sl


def get_swing_high_validity(high_series, window=5):
    """
    現在アクティブな Swing High ライン（get_latest_swing_high の値）が、
    「形成されてから一度も、それより高いローソク足に破られていないか」を判定する。

    Swing High が確定してから window 本の間は定義上必ず安全（それより高い足は
    存在し得ない）だが、それ以降、新しい Swing High が確定するまでの間に
    それより高いローソク足が出現した場合は「無効（invalidated）」となり、
    新たな Swing High が確定するまで False を返し続ける。

    Returns
    -------
    bool の numpy 配列（True = 現在のラインはまだ一度も破られていない ＝ 有効）
    """
    highs = pd.Series(high_series).values
    n = len(highs)
    marks = get_swing_high_marks(highs, window)

    valid = np.zeros(n, dtype=bool)
    current_idx = -1
    current_val = np.nan
    invalidated = False

    for t in range(n):
        check_idx = t - window
        if check_idx >= 0 and not np.isnan(marks[check_idx]):
            # 新しい Swing High が確定 → ラインを更新し、無効フラグをリセット
            current_idx = check_idx
            current_val = marks[check_idx]
            invalidated = False

        if current_idx >= 0 and not invalidated and highs[t] > current_val:
            invalidated = True

        valid[t] = (current_idx >= 0) and not invalidated

    return valid


def get_swing_low_validity(low_series, window=5):
    """
    get_swing_high_validity の Swing Low 版。
    現在アクティブな Swing Low ラインが、形成されてから一度も、
    それより安いローソク足に破られていないかを判定する。
    """
    lows = pd.Series(low_series).values
    n = len(lows)
    marks = get_swing_low_marks(lows, window)

    valid = np.zeros(n, dtype=bool)
    current_idx = -1
    current_val = np.nan
    invalidated = False

    for t in range(n):
        check_idx = t - window
        if check_idx >= 0 and not np.isnan(marks[check_idx]):
            current_idx = check_idx
            current_val = marks[check_idx]
            invalidated = False

        if current_idx >= 0 and not invalidated and lows[t] < current_val:
            invalidated = True

        valid[t] = (current_idx >= 0) and not invalidated

    return valid


def calculate_atr_ema(high, low, close, period=14, price_decimals=5):
    """
    EMA を使用した ATR (Average True Range) の計算
    """
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean().round(price_decimals)
    return atr.values


def compute_signals(signal_df, window=18, atr_period=18, price_decimals=5):
    """
    判定足（既定は4時間足）のOHLCから SwingHigh/Low ラインと有効性フラグ、ATR を計算し、
    「1本分シフト」した DataFrame を返す。

    シフトする理由: 例えば13:00-17:00の判定足の最中（1分足）に、その判定足が確定して
    初めて分かる情報（その足の最終High/Lowを踏まえたSwingHigh等）を
    使ってしまうと未来を覗き見ることになる（ルックアヘッド）。
    そのため「1本前の判定足までの確定情報」を、次の判定足の間ずっと使う形にする。

    Returns
    -------
    DataFrame（判定足と同じインデックス）: 列 SH, SL, SHValid, SLValid, ATR
    """
    high = signal_df["High"].values
    low = signal_df["Low"].values
    close = signal_df["Close"].values

    signals = pd.DataFrame({
        "SH": get_latest_swing_high(high, window),
        "SL": get_latest_swing_low(low, window),
        "SHValid": get_swing_high_validity(high, window).astype(float),
        "SLValid": get_swing_low_validity(low, window).astype(float),
        "ATR": calculate_atr_ema(high, low, close, atr_period, price_decimals),
    }, index=signal_df.index)

    # ルックアヘッド防止のため1本分シフト（当該足の分は1本前の終値時点の情報になる）
    signals = signals.shift(1)
    return signals


def map_signals_to_1min(df_1min, signals, session_start_hour=17, signal_hours=4):
    """
    判定足ベースで計算済みの（1本シフト済み）シグナルを、1分足の各バーに
    対応する判定足のラベルで引き当てて付与する。

    resample_to_signal_bars() と全く同じ区切り（get_signal_bar_label）を
    使うため、判定足と1分足の対応が確実にズレない。

    Returns
    -------
    df_1min に SignalSH / SignalSL / SignalSHValid / SignalSLValid / SignalATR
    列を追加した DataFrame（元の Open/High/Low/Close/Volume はそのまま）
    """
    bar_label = get_signal_bar_label(df_1min.index, session_start_hour, signal_hours)
    mapped = signals.reindex(bar_label)

    out = df_1min.copy()
    out["SignalSH"] = mapped["SH"].values
    out["SignalSL"] = mapped["SL"].values
    out["SignalSHValid"] = mapped["SHValid"].values
    out["SignalSLValid"] = mapped["SLValid"].values
    out["SignalATR"] = mapped["ATR"].values
    return out


def swing_marks_as_dash(marks_func, series, window=5, half_width=1):
    """
    【チャート表示専用】Swing Point を、丸マーカーではなく「ピボット足を中心とした
    短い水平線（前後 half_width 本ぶん）」として描画するための配列を返す。

    丸マーカー（scatter=True）の半径は価格の単位（データ座標）で描画されるため、
    EURUSDのように価格が1前後の通貨では半径が価格帯より大きくなり、縦軸が
    0.6〜1.4のように広がってローソク足が潰れてしまう。線なら自分の値の範囲しか
    使わないので、この問題が起きない。取引判定には使わない（表示のみ）。
    """
    marks = marks_func(series, window)
    out = marks.copy()
    for i in np.where(~np.isnan(marks))[0]:
        out[max(i - half_width, 0): i + half_width + 1] = marks[i]
    return out


class SwingBreakoutStrategy(Strategy):
    # 既定値。実行時は bt.run(...) で Config の値に上書きされる
    window = 18           # Swing Point の前後判定期間（判定足の本数。4時間足なら「前後18本」＝約3日）
    atr_period = 18       # ATR 期間（判定足の本数）
    sl_atr_multiplier = 1.0
    tp_atr_multiplier = 1.5
    price_decimals = 5
    trade_size = 1
    swing_point_style = "dash"  # "dash": 短い水平線 / "circle": 丸マーカー（価格が1前後の通貨では縦軸が潰れる）

    def init(self):
        """
        インジケーターの設定
        """
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        # A. チャート上に Swing Point を描画
        if self.swing_point_style == "circle":
            # 丸点（従来表示）。scatterの半径は価格単位のため、価格が1前後の通貨では
            # 縦軸が広がってローソク足が潰れる点に注意
            self.sh_mark = self.I(
                get_swing_high_marks, high, self.window,
                overlay=True, scatter=True, name="Swing High Point"
            )
            self.sl_mark = self.I(
                get_swing_low_marks, low, self.window,
                overlay=True, scatter=True, name="Swing Low Point"
            )
        else:
            # 短い水平線（既定）。通貨の価格水準に関係なく縦軸が潰れない
            self.sh_mark = self.I(
                swing_marks_as_dash, get_swing_high_marks, high, self.window,
                overlay=True, name="Swing High Point"
            )
            self.sl_mark = self.I(
                swing_marks_as_dash, get_swing_low_marks, low, self.window,
                overlay=True, name="Swing Low Point"
            )

        # B. 現在有効な参照ライン
        self.latest_sh = self.I(
            get_latest_swing_high, high, self.window,
            overlay=True, name="Active Swing High Line"
        )
        self.latest_sl = self.I(
            get_latest_swing_low, low, self.window,
            overlay=True, name="Active Swing Low Line"
        )

        # B'. 「形成後、一度もそのラインを超えるローソク足が出ていないか」の判定
        # チャートを煩雑にしないよう非表示にする。
        # 注意: plot=False だけでは不十分。overlay を省略すると backtesting.py が
        # 「値が終値の0.6〜1.4倍に収まる割合」で自動判定するため、終値が1前後の通貨
        # （EURUSD等）では 0/1 のフラグが価格チャートに重ねて描画され、縦軸が0まで
        # 広がってしまう。overlay=False を明示して判定に依存しないようにする。
        self.sh_valid = self.I(
            get_swing_high_validity, high, self.window,
            plot=False, overlay=False, name="SH Valid"
        )
        self.sl_valid = self.I(
            get_swing_low_validity, low, self.window,
            plot=False, overlay=False, name="SL Valid"
        )

        # C. EMA 平滑化 ATR
        self.atr = self.I(
            calculate_atr_ema, high, low, close, self.atr_period, self.price_decimals,
            overlay=False, name="ATR (EMA)"
        )

    def next(self):
        """
        ローソク足毎のトレード判定
        """
        current_sh = self.latest_sh[-1]
        current_sl = self.latest_sl[-1]
        current_atr = self.atr[-1]
        current_close = self.data.Close[-1]

        # データ準備ができていない場合はスキップ
        if np.isnan(current_sh) or np.isnan(current_sl) or np.isnan(current_atr):
            return

        # TP / SL 幅の計算
        tp_dist = round(current_atr * self.tp_atr_multiplier, self.price_decimals)
        sl_dist = round(current_atr * self.sl_atr_multiplier, self.price_decimals)

        # ポジションを保有していない場合のみ、常に最新ラインに指しておく
        if not self.position:
            # 未約定（Pending）の指値・逆指値注文を全てキャンセル
            for order in self.orders:
                order.cancel()
                
            # 1. Swing High の上に「買い逆指値（Stop Buy）」を常駐させる
            # （現在の終値が Swing High より下にある場合＝まだブレイクしていない場合）
            # かつ、そのSwing Highが形成されてから一度も、それより高いローソク足に
            # 破られていない場合（self.sh_valid）のみエントリー対象とする
            if current_close < current_sh and self.sh_valid[-1]:
                entry_price = round(current_sh, self.price_decimals)
                self.buy(
                    size=self.trade_size,
                    stop=entry_price,
                    tp=round(entry_price + tp_dist, self.price_decimals),
                    sl=round(entry_price - sl_dist, self.price_decimals)
                )

            # 2. Swing Low の下に「売り逆指値（Stop Sell）」を常駐させる
            # （現在の終値が Swing Low より上にある場合＝まだブレイクダウンしていない場合）
            # かつ、そのSwing Lowが形成されてから一度も、それより安いローソク足に
            # 破られていない場合（self.sl_valid）のみエントリー対象とする
            if current_close > current_sl and self.sl_valid[-1]:
                entry_price = round(current_sl, self.price_decimals)
                self.sell(
                    size=self.trade_size,
                    stop=entry_price,
                    tp=round(entry_price - tp_dist, self.price_decimals),
                    sl=round(entry_price + sl_dist, self.price_decimals)
                )


class SwingBreakoutStrategy1Min(Strategy):
    """
    SwingHigh/Low・ATRの「判定」は判定足（signal_hours 時間足。既定は4時間足）ベース
    （前後window本）のまま、実際のエントリー・利確・損切りの「執行」は1分足のOHLCで行う版。

    main() 側で事前に compute_signals() → map_signals_to_1min() により、判定足シグナル
    （1本前の判定足の終値時点までの確定情報のみ・ルックアヘッド無し）を
    1分足の各バーに割り当てた列（SignalSH/SignalSL/SignalSHValid/SignalSLValid/
    SignalATR）を self.data に持たせた状態で使う。

    これにより、同じ判定足の値幅の中でTPとSLの両方に触れるケースでも、
    実際に「どちらのローソク足（分）で先に触れたか」を時系列的に正しく判定できる
    （判定足だけの検証では原理的に区別できなかった問題が解消される）。
    """
    sl_atr_multiplier = 1.0
    tp_atr_multiplier = 1.5
    price_decimals = 5
    trade_size = 1

    def init(self):
        # 判定足ベースで計算済み・1分足にマッピング済みの値をそのまま指標として使う
        # （plot用に self.I() でラップ。中身は既に計算済みなので関数は素通しするだけ）
        self.latest_sh = self.I(lambda: self.data.SignalSH, overlay=True, name="Active Swing High Line (Signal)")
        self.latest_sl = self.I(lambda: self.data.SignalSL, overlay=True, name="Active Swing Low Line (Signal)")
        # 0/1のフラグは非表示。overlay省略だと終値が1前後の通貨で価格チャートに
        # 重ねて描画されてしまうため overlay=False を明示（詳細は上のクラスのコメント参照）
        self.sh_valid = self.I(lambda: self.data.SignalSHValid, plot=False, overlay=False, name="SH Valid")
        self.sl_valid = self.I(lambda: self.data.SignalSLValid, plot=False, overlay=False, name="SL Valid")
        self.atr = self.I(lambda: self.data.SignalATR, overlay=False, name="ATR (Signal, EMA)")

    def next(self):
        """
        1分足のローソク足毎のトレード判定
        （current_sh/sl/atrの値自体は判定足の1本を通して一定＝1本前の判定足終値時点のシグナル）
        """
        current_sh = self.latest_sh[-1]
        current_sl = self.latest_sl[-1]
        current_atr = self.atr[-1]
        current_close = self.data.Close[-1]

        # データ準備ができていない場合はスキップ
        if np.isnan(current_sh) or np.isnan(current_sl) or np.isnan(current_atr):
            return

        # TP / SL 幅の計算（判定足ATRベース）
        tp_dist = round(current_atr * self.tp_atr_multiplier, self.price_decimals)
        sl_dist = round(current_atr * self.sl_atr_multiplier, self.price_decimals)

        # ポジションを保有していない場合のみ、常に最新ラインに指しておく
        if not self.position:
            for order in self.orders:
                order.cancel()

            if current_close < current_sh and self.sh_valid[-1]:
                entry_price = round(current_sh, self.price_decimals)
                self.buy(
                    size=self.trade_size,
                    stop=entry_price,
                    tp=round(entry_price + tp_dist, self.price_decimals),
                    sl=round(entry_price - sl_dist, self.price_decimals)
                )

            if current_close > current_sl and self.sl_valid[-1]:
                entry_price = round(current_sl, self.price_decimals)
                self.sell(
                    size=self.trade_size,
                    stop=entry_price,
                    tp=round(entry_price - tp_dist, self.price_decimals),
                    sl=round(entry_price + sl_dist, self.price_decimals)
                )


def compute_extra_trade_stats(trades):
    """
    取引履歴（stats._trades）から、backtesting.py 標準の Stats に無い指標を計算する。

    - Avg. Win [$]            : 勝ちトレード（PnL > 0）の平均損益
    - Avg. Loss [$]           : 負けトレード（PnL < 0）の平均損益（マイナスの値）
    - Payoff Ratio            : 平均勝ち額 / 平均負け額（絶対値）。勝ちか負けが0件なら NaN
    - Max. Consecutive Losses : 最大連敗数（決済順に並べ、PnL < 0 が連続した最大長。
                                PnL >= 0 のトレードで連敗は途切れる）

    PnL は backtesting.py の値（コミッション控除後）をそのまま使う。
    """
    nan = float("nan")
    if trades is None or len(trades) == 0:
        return {
            "Avg. Win [$]": nan,
            "Avg. Loss [$]": nan,
            "Payoff Ratio": nan,
            "Max. Consecutive Losses": 0,
        }

    # 決済順（同時決済は建玉時刻順）に並べて連敗を数える
    ordered = trades.sort_values(["ExitTime", "EntryTime"], kind="stable")
    pnl = ordered["PnL"].to_numpy(dtype=float)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) else nan
    avg_loss = float(losses.mean()) if len(losses) else nan
    payoff = avg_win / abs(avg_loss) if len(wins) and len(losses) else nan

    max_streak = current = 0
    for p in pnl:
        current = current + 1 if p < 0 else 0
        max_streak = max(max_streak, current)

    return {
        "Avg. Win [$]": avg_win,
        "Avg. Loss [$]": avg_loss,
        "Payoff Ratio": payoff,
        "Max. Consecutive Losses": int(max_streak),
    }


def add_extra_stats(stats):
    """
    bt.run() の結果（stats）に compute_extra_trade_stats() の指標を追加した Series を返す。
    標準の指標（Kelly Criterion まで）の直後に追加し、`_strategy` / `_equity_curve` /
    `_trades` などの内部項目は末尾にそのまま残す（stats._trades 等の既存の使い方は変わらない）。
    """
    extras = pd.Series(compute_extra_trade_stats(stats["_trades"]), dtype=object)
    public = stats[[i for i in stats.index if not str(i).startswith("_")]]
    private = stats[[i for i in stats.index if str(i).startswith("_")]]
    return pd.concat([public, extras, private])


def _format_stat_value(value):
    """Stats 表示用の値の文字列化（浮動小数は有効数字6桁）"""
    if isinstance(value, (float, np.floating)):
        return "NaN" if np.isnan(value) else f"{value:.6g}"
    return str(value)


def append_stats_to_html(html_path, stats, title="Stats"):
    """
    出力済みの HTML チャートの最下部（</body> の直前）に Stats の表を追記する。
    `_` で始まる内部項目（_strategy / _equity_curve / _trades）は表示しない。
    """
    rows = "\n".join(
        f"<tr><th>{escape(str(name))}</th><td>{escape(_format_stat_value(value))}</td></tr>"
        for name, value in stats.items()
        if not str(name).startswith("_")
    )
    block = f"""
<div id="bt-stats" style="max-width:760px;margin:24px auto 48px;padding:0 12px;
     font-family:system-ui,-apple-system,'Segoe UI','Hiragino Sans','Yu Gothic',sans-serif;color:#222;">
  <h2 style="font-size:18px;margin:0 0 8px;">{escape(title)}</h2>
  <table style="border-collapse:collapse;width:100%;font-size:13px;">
    <style>
      #bt-stats th, #bt-stats td {{ border-bottom:1px solid #e3e3e3; padding:4px 10px; }}
      #bt-stats th {{ text-align:left; font-weight:600; width:55%; }}
      #bt-stats td {{ text-align:right; font-variant-numeric:tabular-nums; }}
      #bt-stats tr:hover {{ background:#f6f8fa; }}
    </style>
    {rows}
  </table>
</div>
"""
    path = Path(html_path)
    content = path.read_text(encoding="utf-8")
    idx = content.rfind("</body>")
    content = content[:idx] + block + content[idx:] if idx != -1 else content + block
    path.write_text(content, encoding="utf-8")


def run_sl_tp_grid(bt, fixed_params):
    """
    損切り幅（SL）と利確幅（TP）のATR倍率の全組み合わせを bt.run() で順に検証し、
    主要な成績指標を1つの表にまとめて表示・CSV保存する。

    bt           : 検証データ・Strategy・手数料などを設定済みの Backtest オブジェクト（使い回す）
    fixed_params : SL/TP倍率以外で bt.run() に渡す固定パラメータ（dict）

    倍率の候補は CFG.grid_sl_values / CFG.grid_tp_values、出力先は CFG.grid_csv_filename。
    組み合わせは「利確幅 > 損切り幅（TP倍率 > SL倍率）」のものだけを検証する（同じ倍率は除外）。

    表の指標:
      トレード数 / 勝率 / 収益率 / 最大DD は backtesting.py の Stats の値。
      総利益・総損失・プロフィットファクターは、取引ごとの PnL（コミッション控除後）から計算する
      （総損失はマイナスの値。負けが0件のときプロフィットファクター・ペイオフレシオは NaN）。
      平均利益・平均損失・ペイオフレシオは compute_extra_trade_stats() の定義と同じ。
    """
    all_count = len(CFG.grid_sl_values) * len(CFG.grid_tp_values)
    combos = [(sl, tp) for sl in CFG.grid_sl_values for tp in CFG.grid_tp_values if tp > sl]
    total = len(combos)
    if total == 0:
        print("❌ 検証する組み合わせがありません。grid_sl_values / grid_tp_values が空か、"
              "利確幅 > 損切り幅（TP倍率 > SL倍率）となる組み合わせがありません。")
        return

    print(f"\n================ SL×TP 組み合わせ検証"
          f"（TP倍率 > SL倍率 の {total} 通り / 全 {all_count} 通り中） ================")

    rows = []
    other_warnings = {}  # 同一バー内SL/TP警告以外の警告は、繰り返さず最後に1回だけ出す
    started = time.time()
    for i, (sl, tp) in enumerate(combos, 1):
        run_started = time.time()

        # 「同じバー内でSL/TPが約定する」旨の警告は、組み合わせごとに大量に出て表が
        # 読めなくなるため、ここでは件数だけ数えて進捗行に出す（それ以外の警告は最後に1回ずつ出す）
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stats = bt.run(
                sl_atr_multiplier=sl,
                tp_atr_multiplier=tp,
                price_decimals=CFG.price_decimals,
                trade_size=CFG.size,
                **fixed_params,
            )
        same_bar_warnings = 0
        for w in caught:
            if "contingent SL/TP" in str(w.message):
                same_bar_warnings += 1
            else:
                other_warnings.setdefault((str(w.message), w.category), w)

        trades = stats["_trades"]
        pnl = trades["PnL"].to_numpy(dtype=float) if len(trades) else np.array([], dtype=float)
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        gross_profit = float(wins.sum())
        gross_loss = float(losses.sum())  # マイナスの値
        extra = compute_extra_trade_stats(trades)

        # 組み合わせごとの取引履歴を保存（単発検証の CSV 出力と同じ丸め・形式）
        if CFG.grid_save_trades and len(trades):
            trades_dir = Path(CFG.grid_trades_dir)
            trades_dir.mkdir(parents=True, exist_ok=True)
            out_trades = trades.copy()
            out_trades["PnL"] = out_trades["PnL"].round(CFG.price_decimals)
            out_trades["ReturnPct"] = out_trades["ReturnPct"].round(8)
            out_trades.to_csv(
                trades_dir / f"trades_SL{sl}_TP{tp}.csv",
                index=False,
                float_format=f"%.{CFG.price_decimals}f",
            )

        rows.append({
            "SL倍率": sl,
            "TP倍率": tp,
            "トレード数": int(stats["# Trades"]),
            "勝率 [%]": stats["Win Rate [%]"],
            "収益率 [%]": stats["Return [%]"],
            "総利益 [$]": gross_profit,
            "総損失 [$]": gross_loss,
            "プロフィットファクター": gross_profit / abs(gross_loss) if len(losses) else float("nan"),
            "平均利益 [$]": extra["Avg. Win [$]"],
            "平均損失 [$]": extra["Avg. Loss [$]"],
            "ペイオフレシオ": extra["Payoff Ratio"],
            "最大DD [%]": stats["Max. Drawdown [%]"],
        })

        now = time.time()
        remaining = (now - started) / i * (total - i)
        note = f" / 同一バー内SL/TP警告 {same_bar_warnings} 件" if same_bar_warnings else ""
        print(f"[{i}/{total}] SL×{sl} / TP×{tp}: トレード {int(stats['# Trades'])} 件, "
              f"収益率 {stats['Return [%]']:.2f}%（{now - run_started:.0f}秒{note}）"
              f"  残り約 {remaining / 60:.1f} 分")

    for w in other_warnings.values():
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    table = pd.DataFrame(rows)
    table.round(4).to_csv(CFG.grid_csv_filename, index=False, encoding="utf-8-sig")

    print("\n================ SL×TP 組み合わせ表 ================")
    with pd.option_context("display.unicode.east_asian_width", True,
                           "display.width", 250, "display.max_columns", None):
        print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\n💾 組み合わせ表を CSV に出力しました: {CFG.grid_csv_filename}")
    if CFG.grid_save_trades:
        print(f"💾 組み合わせごとの取引履歴を出力しました: {CFG.grid_trades_dir}/ （trades_SL◯_TP◯.csv）")
    print("※ 総利益・総損失・プロフィットファクターは PnL（コミッション控除後）から計算しています。")
def main():
    # histData フォルダに GMOクリック証券からダウンロードしたZIP（例: USDJPY_202608.zip,
    # USDJPY_202609.zip, ...）を解凍せずそのまま複数まとめて置いてください。
    # フォルダ内の全ZIPの中身を自動的に結合して読み込みます。
    if CFG.signal_hours <= 0 or 24 % CFG.signal_hours != 0:
        print(f"❌ signal_hours は 24 の約数（1, 2, 3, 4, 6, 8, 12, 24）を指定してください: {CFG.signal_hours}")
        return

    if CFG.grid_enabled:
        print(f"ℹ️ 実行モード: SL×TP グリッド検証（SL候補 {CFG.grid_sl_values} × TP候補 {CFG.grid_tp_values}）"
              f" ※チャートは作成しません")
    else:
        print("ℹ️ 実行モード: 単発検証（チャート・取引履歴CSVを出力）"
              " ※SL×TP の組み合わせ表を出すには CFG に grid_enabled=True を指定してください")

    try:
        df_1min, avg_relative_spread = load_gmo_click_1min_data(
            CFG.data_path,
            price_side=CFG.price_side,
            price_decimals=CFG.price_decimals,
        )
        signal_df = resample_to_signal_bars(
            df_1min,
            session_start_hour=CFG.session_start_hour,
            signal_hours=CFG.signal_hours,
        )
    except Exception as e:
        print(e)
        return

    # 仲値でOHLCを構成しているため、往復のスプレッドコストを commission として計上する。
    # backtesting.py の commission はエントリー・イグジットの両方で課金されるため、
    # 平均相対スプレッドの半分を比率として使い、約定ごとの現金手数料に直して丸める。
    # これにより「仲値ベース + 片道スプレッド相当」を近似しつつ、Commission/PnL の桁を抑える。
    commission_rate = avg_relative_spread / 2

    def commission_func(order_size, price):
        return round(abs(order_size) * price * commission_rate, CFG.price_decimals)

    if CFG.execution_timeframe == "signal":
        # ===== 判定・執行とも判定足（4時間足）ベース =====
        bar_trading_day = signal_bar_trading_day(signal_df.index, CFG.session_start_hour)
        full_start, full_end = bar_trading_day.min(), bar_trading_day.max()
        start_ts = pd.Timestamp(CFG.start_date) if CFG.start_date else full_start
        end_ts = pd.Timestamp(CFG.end_date) if CFG.end_date else full_end
        df = signal_df.loc[(bar_trading_day >= start_ts) & (bar_trading_day <= end_ts)]

        if df.empty:
            print(
                f"❌ 指定した検証期間（{CFG.start_date or '指定なし'} ～ {CFG.end_date or '指定なし'}）に"
                f"該当するデータがありません。\n"
                f"   読み込んだデータの範囲: {full_start.strftime('%Y-%m-%d')} ～ {full_end.strftime('%Y-%m-%d')}"
            )
            return

        print(
            f"✅ 検証期間（{CFG.signal_hours}時間足ベース）: {df.index.min().strftime('%Y-%m-%d %H:%M')} ～ "
            f"{df.index.max().strftime('%Y-%m-%d %H:%M')}（{len(df):,} 本 / "
            f"読み込み全体: {full_start.strftime('%Y-%m-%d')} ～ {full_end.strftime('%Y-%m-%d')}）"
        )

        bt = Backtest(df, SwingBreakoutStrategy, cash=CFG.cash, commission=commission_func, margin=CFG.margin, exclusive_orders=False)
        if CFG.grid_enabled:
            run_sl_tp_grid(bt, dict(
                window=CFG.window,
                atr_period=CFG.atr_period,
                swing_point_style=CFG.swing_point_style,
            ))
            return
        print(f"\n================ バックテスト実行中（{CFG.signal_hours}時間足ベース） ================")
        stats = bt.run(
            window=CFG.window,
            atr_period=CFG.atr_period,
            sl_atr_multiplier=CFG.sl_atr_multiplier,
            tp_atr_multiplier=CFG.tp_atr_multiplier,
            price_decimals=CFG.price_decimals,
            trade_size=CFG.size,
            swing_point_style=CFG.swing_point_style,
        )

    else:
        # ===== SwingHigh/Low・ATRの判定は判定足（4時間足）ベース、執行は1分足ベース =====
        # 判定足の全期間（絞り込み前）でシグナルを計算しておくことで、
        # 検証期間の開始直後でも前後window本分の助走情報を失わないようにする
        signals = compute_signals(
            signal_df,
            window=CFG.window,
            atr_period=CFG.atr_period,
            price_decimals=CFG.price_decimals,
        )
        df_1min_ext = map_signals_to_1min(
            df_1min, signals,
            session_start_hour=CFG.session_start_hour,
            signal_hours=CFG.signal_hours,
        )

        # 検証期間の指定（CFG.start_date / CFG.end_date）で、該当する「取引日」の
        # 1分足バーだけに絞り込む（NY 17:00 区切りの取引日ラベルで判定）
        trading_day = get_trading_day_label(df_1min_ext.index, CFG.session_start_hour)
        full_start, full_end = trading_day.min(), trading_day.max()
        start_ts = pd.Timestamp(CFG.start_date) if CFG.start_date else full_start
        end_ts = pd.Timestamp(CFG.end_date) if CFG.end_date else full_end
        df = df_1min_ext.loc[(trading_day >= start_ts) & (trading_day <= end_ts)]

        if df.empty:
            print(
                f"❌ 指定した検証期間（{CFG.start_date or '指定なし'} ～ {CFG.end_date or '指定なし'}）に"
                f"該当するデータがありません。\n"
                f"   読み込んだデータの範囲: {full_start.strftime('%Y-%m-%d')} ～ {full_end.strftime('%Y-%m-%d')}"
            )
            return

        print(
            f"✅ 検証期間（判定={CFG.signal_hours}時間足 / 執行=1分足）: "
            f"{get_trading_day_label(df.index, CFG.session_start_hour).min().strftime('%Y-%m-%d')} ～ "
            f"{get_trading_day_label(df.index, CFG.session_start_hour).max().strftime('%Y-%m-%d')}"
            f"（1分足 {len(df):,} 本 / 読み込み全体: "
            f"{full_start.strftime('%Y-%m-%d')} ～ {full_end.strftime('%Y-%m-%d')}）"
        )

        bt = Backtest(df, SwingBreakoutStrategy1Min, cash=CFG.cash, commission=commission_func, margin=CFG.margin, exclusive_orders=False)
        if CFG.grid_enabled:
            run_sl_tp_grid(bt, {})
            return
        print(f"\n================ バックテスト実行中（判定={CFG.signal_hours}時間足 / 執行=1分足） ================")
        stats = bt.run(
            sl_atr_multiplier=CFG.sl_atr_multiplier,
            tp_atr_multiplier=CFG.tp_atr_multiplier,
            price_decimals=CFG.price_decimals,
            trade_size=CFG.size,
        )
    
    # backtesting.py 標準の Stats に、Payoff Ratio・最大連敗数などを追加
    stats = add_extra_stats(stats)

    # ターミナルに主要結果を出力
    print(stats)

    # 年ごとの集計を作成
    trades = stats._trades.copy()

    # 日時を datetime に変換
    trades["EntryTime"] = pd.to_datetime(trades["EntryTime"])
    trades["ExitTime"] = pd.to_datetime(trades["ExitTime"])

    # 年ごとに分類
    trades["Year"] = trades["EntryTime"].dt.year

    yearly = (
        trades.groupby("Year")
        .agg(
            Trades=("EntryTime", "count"),
            NetPL=("PnL", "sum"),
            WinRate=("PnL", lambda s: (s > 0).mean()),
            AvgPL=("PnL", "mean"),
        )
        .sort_index()
    )

    print(yearly)


    # 取引履歴を CSV に出力
    trades_df = stats.get('_trades')
    if trades_df is not None and not trades_df.empty:
        trades_df["PnL"] = trades_df["PnL"].round(CFG.price_decimals)
        trades_df["ReturnPct"] = trades_df["ReturnPct"].round(8)
        trades_df.to_csv(
            CFG.csv_filename,
            index=False,
            float_format=f"%.{CFG.price_decimals}f",
        )

        print(f"\n💾 取引履歴を CSV に出力しました: {CFG.csv_filename}")
    else:
        print("\n⚠️ 取引履歴がありません。CSV出力はスキップしました。")

    # 取引履歴の確認ログを表示
    trade_count = stats['# Trades']
    print(f"\n--------------------------------------------------")
    print(f"📊 総取引回数 (# Trades): {trade_count} 件")
    
    if trade_count == 0:
        print("⚠️ 注意: トレードが1件も発生していません。エントリー条件等を見直してください。")
    else:
        print("\n【直近の取引履歴 (先頭5件)】")
        print(stats['_trades'].head())
    print(f"--------------------------------------------------")

    # インタラクティブチャートの生成とブラウザ表示
    # chart_timeframe="signal" なら、バックテスト自体は1分足精度で実行しつつ、
    # チャート表示だけ判定足（4時間足）のローソク足にまとめる（表示専用でstatsには影響しない）
    plot_bt = bt
    n_bars = len(df)

    if CFG.execution_timeframe == "signal":
        # 既に判定足なのでそのまま全描画
        chart_resample = False
    elif CFG.chart_timeframe == "signal":
        # bt.plot(resample="4h") は「0:00起点」で集約するため、NY 17:00起点の判定足とは
        # 境界がズレる。表示専用に、集約の境界が判定足の境界と一致するよう時刻をずらした
        # コピーで同じバックテストを再実行し、そのチャートで判定足に集約する。
        # （約定ロジックは同一なので結果は同じ。stats/CSVは上で出力済みの実時刻のもの）
        chart_df = df.copy()
        chart_df.index = make_chart_index_for_signal_bars(
            df.index, CFG.session_start_hour, CFG.signal_hours
        )
        plot_bt = Backtest(chart_df, SwingBreakoutStrategy1Min, cash=CFG.cash,
                           commission=commission_func, margin=CFG.margin, exclusive_orders=False)
        plot_stats = plot_bt.run(sl_atr_multiplier=CFG.sl_atr_multiplier, tp_atr_multiplier=CFG.tp_atr_multiplier, price_decimals=CFG.price_decimals, trade_size=CFG.size)
        if plot_stats["# Trades"] != stats["# Trades"]:
            print("⚠️ チャート用の再実行結果とトレード数が一致しません。チャートの表示内容にご注意ください。")
        chart_resample = f"{CFG.signal_hours}h"
    else:
        # "native": 本数が多い場合のみ自動間引き（20,000本以下はそのまま全描画）
        chart_resample = False if n_bars <= 20_000 else True
        if chart_resample is True:
            print(f"ℹ️ ローソク足が{n_bars:,}本と多いため、チャートはbokehの自動リサンプル表示にします"
                  f"（全トレードマークを見たい場合は検証期間を絞るか、chart_timeframe='signal'にしてください）。")

    print(f"\n📊 チャートを生成中... ({CFG.html_filename})")
    # open_browser=False で保存だけ行い、Stats を追記してから自前でブラウザを開く
    # （open_browser=True だと追記前のHTMLがブラウザで開かれてしまうため）
    plot_bt.plot(
        filename=CFG.html_filename,
        open_browser=False,
        resample=chart_resample,
        # GMOクリック証券のFXデータには出来高が無く Volume を 0 で埋めているため、
        # 出来高パネルを出すと縦軸が-1〜1に潰れた意味のない線が表示される。非表示にする。
        plot_volume=False,
    )

    # HTMLの一番下に Stats（Payoff Ratio・最大連敗数を含む）を追記
    append_stats_to_html(CFG.html_filename, stats)
    print(f"📝 Stats を HTML の最下部に追記しました: {CFG.html_filename}")

    # bokeh の show()（＝従来の open_browser=True）と全く同じ方法でブラウザを開く。
    # 一度 Path.as_uri()（file:///C:/... 形式）で開くようにしたところ、Windowsで既定の
    # ブラウザ（Chrome）ではなくEdgeで開かれてしまったため、bokeh と同じ
    # "file://" + 絶対パス を、新しいタブ指定（new=2）で webbrowser に渡す形に戻している。
    try:
        webbrowser.open("file://" + os.path.abspath(CFG.html_filename), new=2)
    except Exception:
        pass  # ブラウザを開けない環境（サーバー等）では無視


if __name__ == '__main__':
    main()