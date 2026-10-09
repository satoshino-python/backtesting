# 検証用コードの一覧と設定ガイド

このフォルダの各スクリプトの「売買ルール」「設定できる項目」「出力」をまとめたもの（2026-10-05 時点のコードに基づく）。
ダウ理論トレンド判定の詳細仕様は [DOW_TREND_SPEC.md](DOW_TREND_SPEC.md) を参照。

## 1. ファイル一覧

| ファイル | 役割 | 判定足 | 決済 | 枚数 |
|---|---|---|---|---|
| `main_D1.py` | 最初の版。日足のスイングブレイク | 日足 | ATR倍率の固定SL/TP | 固定（`size`） |
| `main_4H.py` | 判定足を4時間足（任意の時間足）に変更。SL×TPグリッド検証を追加 | 4時間足 | ATR倍率の固定SL/TP | 固定（`size`） |
| `main_4H_fixedSL.py` | **基準となる版**。1回の損失額を固定する枚数計算と建値ストップを追加 | 4時間足 | 固定SL/TP＋建値ストップ | 損失額固定（`risk_pct`） |
| `main_4H_fixedSL_multi.py` | 全通貨ペアを一括検証し、R倍数で合算。1時間足スイングのトレーリングストップも選べる | 4時間足 | 固定SL/TP または トレーリング | 損失額固定 |
| `main_4H_fixedSL_dow.py` | 週足ダウ理論のトレンド方向にだけエントリーするフィルター版 | 4時間足＋週足 | 固定SL/TP＋建値ストップ | 損失額固定 |
| `main_4H_fixedSL_fast_multi.py` | `main_4H_fixedSL_multi.py` の高速版。`fast_engine.py` で全ペアを並列に検証 | 4時間足 | 固定SL/TP または トレーリング | 損失額固定 |
| `fast_engine.py` | 1分足執行版の戦略（固定SL/TP・1時間足トレーリング）を backtesting.py なしで約100倍速く検証するエンジン | 4時間足 | 固定SL/TP または トレーリング | 損失額固定 |
| `compare_fast_engine.py` | `fast_engine.py` と backtesting.py の結果が一致するかを確かめる | – | – | – |
| `dow_trend.py` | ダウ理論のトレンド判定ライブラリ（単体では売買しない） | 任意 | – | – |
| `main_4H_dow_swingExit.py` | 週足ダウ理論フィルター＋4時間足スイングのトレーリングストップ決済。トレンドの質フィルターも選べる | 4時間足＋週足 | 4Hスイングへのトレーリング | 損失額固定 |
| `trend_quality_study.py` | 上の戦略にトレンドの質フィルターを組み合わせ、全ペアで期間を分けて比較 | 4時間足＋週足 | 同上 | 損失額固定 |
| `dow_swing_chart.py` | ダウ理論のスイングとトレンド、トレードを確認するインタラクティブチャート | 1H/4H/D1/W1 | – | – |
| `compare_runs.py` | `multi_results/` の複数の実行結果を並べて比較 | – | – | – |
| `diagnose_trades.py` | 取引履歴CSVを、時間帯・曜日・ATR水準などの切り口で診断 | – | – | – |
| `tests/test_dow_trend.py` | `dow_trend.py` のテスト | – | – | – |
| `build_h4_cache.py` | 1分足を4時間足にまとめて `cache/h4_<ペア>.pkl` に保存（下の2本の下準備） | 4時間足 | – | – |
| `trend_filter_study.py` | `fast_results` の取引に、エントリー直前の方向・強さの指標（ADX・ER・CHOP・移動平均・週足ダウなど）を付けて成績との関係を集計 | 4H/D1/W1 | – | – |
| `filter_rerun.py` | 方向フィルター（`fast_engine` の `AllowLong`/`AllowShort` 列）を入れて全ペアを再検証。結果は `filter_results/<日時>_rerun/` | 4時間足＋日足/週足 | 固定SL/TP と 1Hトレーリング | 損失額固定 |
| `dow_structure_breakout.py` | 4時間足のダウ理論スイングで高値・安値とも切り上げ（切り下げ）なら、直近スイングハイ（ロー）に逆指値。従来版と比べる。結果は `fast_results/<日時>_dow_structure/` | 4時間足 | 固定SL/TP と 1Hトレーリング | 損失額固定 |
| `dow_daily_breakout.py` | 日足ダウ・押し目ブレイク。日足で高値・安値を切り上げて押し目が確定したら直近高値に買い逆指値（売りは逆）、損切りは押し目の安値。専用の自前エンジンで、スイング幅×最小スイング幅×決済の12通りを全ペア検証。結果は `fast_results/<日時>_dow_daily/` | 日足 | 押し目追従 / 利確2R・3R | 損失額固定（1%） |
| `compare_dow_daily.py` | `dow_daily_breakout.py` のエンジンと backtesting.py の結果が一致するかを確かめる | – | – | – |

`main_4H_fixedSL_multi.py` と `main_4H_fixedSL_dow.py` は、`main_4H_fixedSL.py` の関数と Strategy を import して使っている。
**`main_4H_fixedSL.py` を変更すると、この2つの結果も変わる。**
`fast_engine.py` は `SwingBreakoutStrategy1Min`・`move_sl_to_breakeven()`（`main_4H_fixedSL.py`）と
`SwingBreakoutTrail1Min`（`main_4H_fixedSL_multi.py`）の売買ルールを自前で再現しているため、
**これらのルールを変えたら `fast_engine.py` も合わせて直し、`compare_fast_engine.py` で一致を確かめる。**

## 2. 全スクリプト共通の仕組み

### データ
- GMOクリック証券の1分足ヒストリカルデータ（BID/ASK）。月ごとの ZIP を解凍せずに `histData/<通貨ペア>/` に置く。
  - 現在あるフォルダ: `AUDUSD` / `EURUSD` / `GBPUSD` / `USDCHF` / `USDJPY`（2020-01〜2025-12 ほか）、`SP500`（US500、2024-01〜2025-12）
  - CSV の日時の書式は銘柄で違う（FX は `2024/01/15 07:00:00`、US500 は `202401020800`）。どちらも日本時間で、読み込み時に両方に対応している。
  - 2023年6月以前の ZIP に同梱されている `_EX` 付きの別系列 CSV は読み込まない（`main_4H_fixedSL.py` 以降）。
- タイムスタンプは日本時間で記録されているため、NY時間に変換する（サマータイムは自動で考慮）。
- 価格は既定で仲値（`(BID+ASK)/2`）を使う。スプレッドは、データ全体の平均相対スプレッドの半分を
  エントリーとイグジットのそれぞれにコミッションとして課金して近似する（往復でスプレッド1回分）。
- 出来高データは無いため、Volume は 0 で埋める。

### 足の区切り
- 1日の区切りは **NY 17:00**（NYクローズ）。4時間足なら 17-21 / 21-1 / 1-5 / 5-9 / 9-13 / 13-17 時の6本/日。
- 取引日のラベルは、NY 17:00〜翌17:00 の大半が含まれる側の日付（月〜金）。`start_date` / `end_date` はこの取引日で絞り込む。

### 執行の精度
- `execution_timeframe="1min"`（既定・推奨）: スイングと ATR の判定は判定足で行い、エントリー・利確・損切りの約定は1分足で判定する。
  同じ4時間足の中でTPとSLの両方に触れた場合も、どちらが先かを正しく判定できる。
- `"signal"`（D1版では `"daily"`）: 判定も約定も判定足で行う。速いが、足の中の値動きの順番は分からない。

### ルックアヘッド（未来データの参照）防止
- 判定足のシグナルは1本ずらして使う。判定足の途中の1分足では、1本前の判定足の終値時点で確定していた情報だけを使う。
- スイングは前後 `window` 本で判定するため、ピボットの足から `window` 本後に初めて確定する。

## 3. 共通のエントリールール（スイングブレイクアウト）

1. **スイングハイ / スイングロー**: 前後 `window` 本の中で最高値（最安値）の足。同値も含む。
2. **有効なライン**: 直近で確定したスイングハイ（ロー）のうち、確定してから一度もそれを超える高値（下回る安値）が
   出ていないもの。一度抜かれたら、次のスイングが確定するまでそのラインではエントリーしない。
3. **注文**: ポジションが無いとき、毎バーで未約定注文をすべて取り消し、置き直す。
   - 終値 < スイングハイ かつ 有効 → スイングハイに **買い逆指値**
   - 終値 > スイングロー かつ 有効 → スイングローに **売り逆指値**
   - 買いと売りの両方を同時に置くことがある。
4. **利確・損切り**（固定SL/TPの版）:
   - 損切り = エントリー価格 ∓ ATR × `sl_atr_multiplier`
   - 利確 = エントリー価格 ± ATR × `tp_atr_multiplier`
   - ATR は判定足の True Range を EMA（span = `atr_period`）で平滑化したもの。
     `dow_trend.py` で使う ATR はワイルダー平滑化なので、計算方法が違う。

> **注意（コード上の挙動）**: 固定SL/TPの版（D1 / 4H / fixedSL / dow）は、ポジション保有中は注文を置き直さない。
> そのため、エントリーしなかった反対側の逆指値が残り、価格がそこまで動くと既存ポジションの決済として約定する。
> `main_4H_fixedSL_multi.py` の `swing_trail` モードだけは、片方が約定した時点で反対側の注文を取り消す。

## 4. 各スクリプトの詳細

### 4.1 `main_D1.py`（日足版）
- 判定足は日足（NY 17:00 区切り）。`window=5`（前後5日）、`atr_period=14`。
- 枚数は固定で、`size`（既定 10000）を使う。
- 現在の設定: 2021-01-01〜2025-12-31、SL 1.0 × ATR、TP 1.0 × ATR。
- 出力: `trade_history.csv`、`swing_breakout_chart.html`（チャートの最下部に成績表）。
- `chart_timeframe`: `"daily"`（日足にまとめて表示）/ `"native"`。

### 4.2 `main_4H.py`（4時間足版）
- `signal_hours` で判定足の長さを変えられる（24の約数: 1, 2, 3, 4, 6, 8, 12, 24）。`window=18`（4時間足18本 ≒ 3日）、`atr_period=18`。
- 枚数は D1版と同じく固定の `size`。
- **SL×TP グリッド検証**（`grid_enabled=True`）: `grid_sl_values` × `grid_tp_values` のうち、TP倍率 > SL倍率 の組み合わせをすべて検証し、
  `sl_tp_grid.csv` に一覧を出す。組み合わせごとの取引履歴は `grid_trades/trades_SL◯_TP◯.csv`。
- 現在の設定: 2021-01-01〜2025-12-31、SL 1.5、TP 2.5、グリッドなし。

### 4.3 `main_4H_fixedSL.py`（基準版）
`main_4H.py` に次の変更を加えたもの。

- **損失額固定の枚数計算**: 枚数 = 初期資金 × `risk_pct` ÷ 損切り幅。
  - 損切りになると、損失はおおよそ「初期資金 × `risk_pct`」（＝ **1R**）になる。既定は 10,000 × 2% = 200。
  - 初期資金を基準にした固定額なので、複利にはならない。
  - 窓開けでSLを飛び越えた場合は 1R を超えることがある。証拠金が足りない注文は backtesting.py がキャンセルする。
- **建値ストップ**: 含み益が `breakeven_trigger_r` × R に達したら、SLをエントリー価格に移す。`None` で無効。
  - 判定は現在のバーの高値・安値で行い、移動したSLは次のバーから有効になる。到達したのと同じバーで建値まで戻った場合は、当初のSLのまま扱う（保守的）。
- `_EX` 付きの CSV を除外する処理を追加した。
- 成績に「平均勝ち / 平均負け / ペイオフレシオ / 最大連敗」を追加し、HTMLチャートの最下部に表示する。
- 現在の設定: 2021-01-01〜2025-12-31、SL 1.5、TP 2.5、建値ストップ 1.0R（Config の既定値）、グリッドなし。

### 4.4 `main_4H_fixedSL_multi.py`（複数通貨ペア一括検証）
- `histData/` の下にある通貨ペアのフォルダを順番に検証する。各ペアは、同じ初期資金を持つ別々の口座として扱う。
  複数ペアの同時保有による証拠金の取り合いは再現しない。
- 損益は通貨ペアごとに決済通貨が違うため、**R倍数**（損益 ÷ 1R）に換算して合算する。
  合計R × `risk_pct` が、初期資金に対する損益率の目安になる（例: +50R × 2% = +100%）。
- `price_decimals` はペア名で自動設定する（〜JPY は3桁、`SP500`/`US500` は2桁、それ以外は5桁）。
- 読み込む ZIP は、検証期間の `WARMUP_MONTHS` ヶ月前（助走期間）から終了月の翌月まで。平均スプレッドは検証期間だけで計算する。

**ファイル冒頭の設定**

| 名前 | 現在の値 | 意味 |
|---|---|---|
| `BASE_CFG` | 2021-01-01〜2025-12-31、SL 1.5、TP 2.5、建値ストップなし | 全ペア共通の `Config` |
| `DATA_ROOT` | `histData` | 通貨ペアのフォルダがある場所 |
| `PAIRS` | `None` | `None` ならフォルダ内の全ペアを検証。例: `("EURUSD", "USDJPY")` |
| `RESULTS_ROOT` / `RUN_LABEL` | `multi_results` / `"trailH1w5_atrSL"` | 結果は `multi_results/<日時>_<RUN_LABEL>/` に保存し、上書きしない |
| `OPEN_CHARTS` | `True` | ペアごとのチャートをブラウザで開く |
| `WARMUP_MONTHS` | `1` | 助走期間（月） |
| `EXIT_MODE` | `"swing_trail"` | `"fixed"`: 固定SL/TP（建値ストップは `BASE_CFG` に従う） / `"swing_trail"`: 下記 |
| `H1_SWING_WINDOW` | `5` | トレーリングに使う1時間足スイングの前後本数 |
| `INITIAL_SL_RULE` | `"atr"` | `swing_trail` の当初SL。`"atr"`: ATR × 倍率 / `"near"`・`"far"`: ATR と直近の1時間足スイングのうち近い方・遠い方 |

**`swing_trail` モード**（`execution_timeframe="1min"` のときだけ使える）
- エントリーは共通ルールと同じ。TP と建値ストップは無い。
- エントリーの後に新しく確定した1時間足スイングがあれば、そこへSLを移す（買いなら Swing Low、有利な方向にだけ動かす）。
- 枚数は、当初SLまでの距離で 1R になるように決める。

**出力**（`multi_results/<日時>_<RUN_LABEL>/`）
- `summary.csv`: ペア別と合計の成績（トレード数、勝率、合計R、平均R、PF、ペイオフ、最大連敗、最大DD[R]、収益率）
- `yearly.csv`: 年別 × ペア別の合計R、トレード数、勝率
- `trades_all.csv`: 全取引（`Pair` 列と `R` 列付き）
- `equity_R_combined.csv`: 全ペア合算の累積R（1時間ごと、含み損益込み）
- `chart_<ペア>.html`: ペアごとのチャート
- `config.json` と `code/`: 実行時の設定とコードのコピー。どの設定とコードで出した結果かを後から辿れる。

**これまでの実行**（`multi_results/` 内）

| フォルダ | 内容 |
|---|---|
| `20261004_2142_BE1.0` | 固定SL/TP＋建値ストップ1R（SP500 なし） |
| `20261004_2215_noBE` | 固定SL/TP、建値ストップなし |
| `20261004_2242_trailH1` | 1時間足トレーリング、当初SL = far |
| `20261004_2252_trailH1_nearSL` | 1時間足トレーリング、当初SL = near |
| `20261004_2307_trailH1w5_atrSL` | 1時間足トレーリング（前後5本）、当初SL = atr |

### 4.5 `main_4H_fixedSL_dow.py`（週足ダウ理論フィルター）
- エントリー・決済・枚数は `main_4H_fixedSL.py` と同じ。週足のトレンドに合わない方向の逆指値だけを取り消す。
- 週足は、NY 17:00 区切りの取引日を金曜日で終わる1週間にまとめたもの。各週には前週末に確定した判定を使う（ルックアヘッド防止）。
- 1回の実行で `compare_modes`（フィルターなし / strict / no_counter）を順番に検証し、比較表を作る。

**`DowFilterConfig` の追加項目**（`Config` の全項目も使える）

| 名前 | 既定 | 意味 |
|---|---|---|
| `dow_n` | 3 | ピボット幅（左右 n 週） |
| `dow_atr_period` | 14 | 週足ATRの期間 |
| `dow_min_swing_atr` | 1.0 | 最小スイング幅（週足ATRの倍数）。0 でフィルターなし |
| `dow_use_wick` | True | True: ヒゲでブレイク判定 / False: 終値で判定 |
| `dow_mode` | `"strict"` | `"strict"`: 上昇なら買いだけ、下降なら売りだけ、レンジは取引なし / `"no_counter"`: トレンドに逆らう方向だけ止める（レンジは両方向） |
| `compare_modes` | `(None, "strict", "no_counter")` | 比較表に並べるモード（`None` ＝ フィルターなし） |
| `result_dir` | `"dow_results"` | 保存先 |

- 現在の設定: EURUSD、2025-01-01〜2025-12-31、SL 1.5、TP 2.5。`OPEN_CHART=True`、`TRADE_CHART=True`。
- 出力（`dow_results/`）: `comparison_<期間>.csv`、`trades_<期間>_<モード>.csv`、`weekly_trend_<期間>.csv`（各週に使った判定）、`chart_<期間>_<dow_mode>.html`（backtesting.py のチャート）、
  `dow_trade_chart_<通貨>_<期間>_<dow_mode>.html`（トレード付きのダウ理論チャート。4.9 参照。`TRADE_CHART=False` で出力しない）。
- `execution_timeframe="1min"` のみ対応。

### 4.6 `dow_trend.py`（トレンド判定ライブラリ）
- `compute_dow_trend(df, DowConfig(...))` は、各バーのトレンド（1=上昇 / 0=レンジ / -1=下降）と、採用したスイングの一覧を返す。
- スイングは左右 n 本より**厳密に**高い（低い）足（同値は不採用）。ATRによる最小幅フィルターがあり、H と L は必ず交互に並ぶ。
- 安値を割ったとき、戻り高値がその前のスイングハイより低ければ下降。上昇も同じ考え方。
- 詳細とテスト結果は [DOW_TREND_SPEC.md](DOW_TREND_SPEC.md)。テストは `python tests/test_dow_trend.py`。

### 4.7 `compare_runs.py`（実行結果の比較）
```
python compare_runs.py                 # multi_results/ 内の全ての実行を比較
python compare_runs.py noBE BE1.0      # フォルダ名にこの文字列を含む実行だけ
```
- 実行ごとに違う設定だけを抜き出して、総合成績とペア別の合計Rを並べる。結果は `multi_results/comparison.csv` に保存する。

### 4.8 `diagnose_trades.py`（取引履歴の診断）
```
python diagnose_trades.py                         # 最新の trade_history_*.csv
python diagnose_trades.py sl1.5_tp2.5             # ファイル名の一部で指定
python diagnose_trades.py SL1.5_TP2.5             # grid_trades/ の中も探す
python diagnose_trades.py sl1.5_tp2.5 --mae-mfe --data-path histData/EURUSD
```
| オプション | 既定 | 意味 |
|---|---|---|
| `--pip-size` | 0.0001 | 1pipの価格幅（USDJPY は 0.01） |
| `--price-tol` | pip-size の1/10 | 決済価格とSL/TPの一致を判定するときの許容幅 |
| `--out-dir` | `diagnosis` | 出力先 |
| `--mae-mfe` | なし | MAE/MFE（保有中の最大逆行・最大順行）も計算する。元の1分足データが必要 |
| `--data-path` | `main_4H.py` の `data_path` | 1分足データの場所 |

- 診断の切り口: 全体成績（期待値・t値）、買い/売り、年、エントリー時間帯、曜日、ATR水準、スイング幅、決済理由（SL/TP/その他）、保有時間、コスト感度、MAE/MFE。
- 出力: `diagnosis/diagnosis_report.txt`、`diagnosis/trades_enriched.csv`。
- 4時間足版（`main_4H.py` / `main_4H_fixedSL.py`、`execution_timeframe="1min"`）の取引履歴の列名（`Entry_ATR (Signal, EMA)` など）を前提にしている。D1版の CSV には対応していない。

### 4.9 `dow_swing_chart.py`（ダウ理論・トレードの確認用チャート）
```
python dow_swing_chart.py                                                   # スイング・トレンド判定だけ
python dow_swing_chart.py --trades dow_results/trades_20210101-20251231_strict.csv   # トレードも表示
```
- 1H / 4H / D1 / W1 を切り替えられるインタラクティブチャート（HTML 1ファイル、外部ライブラリ不要）。ピボット幅・ATR・最小スイング幅をスライダーで変えると、ダウ理論の判定がその場で再計算される。
- `--trades` に `main_4H_fixedSL_dow.py` の取引履歴CSVを渡すと、次の表示が加わる。出力は `dow_results/dow_trade_chart_<通貨>_<期間>_<モード>.html`。
  - エントリー（▲買い / ▼売り）と決済（●、緑=勝ち / 赤=負け）、その間を結ぶ線、SL/TP の点線
  - 背景: 検証で実際に使った週足トレンド（同じフォルダの `weekly_trend_<期間>.csv`）。「表示中の時間足で計算」に切り替えることもできる。
  - エントリーライン: 4時間足の Swing High / Low（`--entry-window`、既定18）。薄い線は「一度抜かれて無効」の区間。1H と 4H だけ表示する。
  - 「前 / 次のトレード」ボタン（キーボードの ← → でも可）。トレードにカーソルを合わせると、エントリー・決済・損益(R)・決済理由を表示する。
- その他のオプション: `--symbol` / `--start` / `--end` / `--weekly-trend` / `--entry-atr` / `--risk`（1R の金額。既定 200）
- スクリプトからは `make_chart(df_1min, path, title, start, end, trades=stats["_trades"], weekly_trend=..., risk=...)` で作れる。
  `main_4H_fixedSL_dow.py` はこれを使って毎回出力している。他の戦略の検証でも使える（週足トレンドを使わない場合は `weekly_trend=None`）。使い方の例は [CLAUDE.md](../CLAUDE.md)。

### 4.10 `main_4H_dow_swingExit.py`（週足ダウ＋4Hスイング決済）
- エントリー: `main_4H_fixedSL_dow.py` と同じ（4時間足 `window`=18 のスイングブレイク、週足ダウ `dow_n`=3、`dow_mode`="strict"）。片方が約定したら反対側の注文は取り消す。
- 決済: TP・建値ストップなし。当初SL＝直近の4時間足スイング（前後 `exit_window`=6 本。買いなら Swing Low）。新しいスイングが確定するたび有利な方向にだけ動かす。
  直近スイングがエントリー価格の反対側に無いときは発注しない。
- 枚数: 当初SLまでの距離で 1R（初期資金 × `risk_pct`）になる枚数。
- トレンドの質フィルター（`None` で無効）: `min_trend_age`（週足判定が同じ向きで続いている週数）、`min_weekly_adx`（週足ADX(14)）、
  `min_h4_er`（4時間足の効率比 `er_period`=18 本。取引方向を正とする）。週足の値は前週末、4時間足は1本前の足で確定したものを使う。
- 出力: `dow_results/<日時>_<run_label>/`（取引履歴、成績、日次資産、使った週足トレンド、設定、トレード付きダウ理論チャート、backtesting.py のチャート）。
- 期末に保有中のトレードは取引履歴に入らないが、資産（収益率）には含み損益として入る。

### 4.11 `trend_quality_study.py`（トレンドの質フィルターの比較）
- `PAIRS` の各ペアで、`GRID`（継続週数 × 週足ADX × 4H効率比）の全組み合わせを実行し、R倍数で合算する。ペアごとに別プロセスで並列実行（`WORKERS`）。
- `IN_SAMPLE`（既定 2021-2023）で条件を選び、`OUT_OF_SAMPLE`（2024-2025）で確かめる。全期間で1回実行し、エントリー日で分けて集計する。
- 出力: `quality_results/<日時>_<RUN_LABEL>/`（`summary.csv`: 組み合わせ別の IS/OOS/全期間の成績とペア別R、`trades_all.csv`、`config.json`、`code/`）。
- 1組み合わせ・1ペアあたり約35秒（5年分）。

### 4.12 `fast_engine.py` / `main_4H_fixedSL_fast_multi.py` / `compare_fast_engine.py`（高速版）
- `fast_engine.run_fast(df, cash=, commission=, margin=, sl_atr_multiplier=, price_decimals=, risk_pct=, ...)` は backtesting.py と同じ stats を返す
  （`_trades`・`_equity_curve` を含む。`_trades` に `Entry_…`/`Exit_…` の指標列は無い）。`df` は `map_signals_to_1min()` で Signal* 列を付けた1分足。
  - `exit_mode="fixed"`（既定）: `tp_atr_multiplier=`・`breakeven_trigger_r=` を渡す。`SwingBreakoutStrategy1Min` と同じ。
  - `exit_mode="swing_trail"`: `initial_sl_rule=` を渡す。`df` に `compute_h1_swings()` の列（H1SH/H1SL/H1SHSeq/H1SLSeq）も必要。
    `SwingBreakoutTrail1Min` と同じ。
- 仕組み: 何も起きない1分足を numpy でまとめて飛ばし、エントリー・SL・TP・建値ストップ・トレーリング・残った反対側の逆指値が
  動くバーだけを backtesting.py 0.6 のブローカーと同じ手順で処理する（窓開けの約定価格、同じバーで SL と TP に触れたら SL 優先、
  エントリーと同じバーの SL/TP の扱い、反対側の逆指値による決済、一部決済、証拠金不足の取消しまで再現）。
- `SwingBreakoutTrail1Min` の注意（backtesting.py の動きをそのまま再現している）: 新しい1時間足スイングが既に終値の反対側にあると
  `trade.close()` を出すが、同じ `next()` の「contingent でない注文の取消し」で消えるため成行決済はされない。
  終値がスイングの内側に戻ったバーで SL が移動する。
- 確認結果（2026-10-06）: 5ペア × 2021〜2025年と SP500（2024-02〜2025-11）で、fixed（建値ストップなし・あり）と
  swing_trail（当初SL atr / near / far）の全組み合わせで、取引履歴・資産曲線・統計値が backtesting.py と完全に一致。
  1ペア5年分で backtesting.py 約150秒 → 約1秒。
- `main_4H_fixedSL_fast_multi.py`: `main_4H_fixedSL_multi.py` と同じ検証（`EXIT_MODE` = `"fixed"` / `"swing_trail"`、`H1_SWING_WINDOW`、
  `INITIAL_SL_RULE`）を、ペアごとに別プロセスで並列実行する（`WORKERS`、既定4。1ペアあたり最大1〜2GBのメモリを使う）。
  集計は multi と同じ（ペア別・総合・年別を R倍数で）。`RUN_LABEL=None` なら決済ルールからフォルダ名を自動で付ける。
  チャートはトレード付きのダウ理論チャートだけを出力する（`MAKE_CHARTS`）。backtesting.py 標準のチャートは出さない。
- 出力: `fast_results/<日時>_<RUN_LABEL>/`（`summary.csv`、`yearly.csv`、`trades_all.csv`、`trades_<ペア>.csv`、`stats_by_pair.csv`、
  `equity_R_combined.csv`、`dow_trade_chart_<ペア>_*.html`、`log_<ペア>.txt`、`config.json`、`code/`）。
- 所要時間: 5ペア × 5年分でチャート込み約36秒（読み込み約10秒・検証約1秒・チャート約8秒 / ペア、4並列）。
- `compare_fast_engine.py`: `python compare_fast_engine.py EURUSD 2024-01-01 2024-12-31 [breakeven_trigger_r=1.0]`、
  `... exit_mode=swing_trail [initial_sl_rule=near] [h1_window=5]` で両方を実行して比べる。
  食い違ったら両方の取引履歴を `fast_compare_results/<日時>_<ペア>/` に保存する。

## 5. `Config` の設定項目（4時間足系の共通項目）

設定は各ファイルの `CFG = Config(...)`（multi は `BASE_CFG`）で上書きする。書かなかった項目は下の既定値になる。

| 項目 | 既定 | 意味 |
|---|---|---|
| `data_path` | `histData/EURUSD` | データのフォルダ（または ZIP） |
| `price_side` | `"mid"` | `"mid"` / `"bid"` / `"ask"` |
| `session_start_hour` | 17 | 足の区切りの起点（NY時間） |
| `signal_hours` | 4 | 判定足の長さ（24の約数） |
| `cash` | 10,000 | 初期資金 |
| `size` | 10000 | 1回の枚数（`main_D1.py` / `main_4H.py` のみ） |
| `risk_pct` | 0.02 | 1回の損失額 ＝ 初期資金 × この値（fixedSL 系のみ） |
| `window` | 18 | スイング判定の前後本数（D1版は5） |
| `atr_period` | 18 | ATR期間（D1版は14） |
| `sl_atr_multiplier` | 1.0 | 損切り幅 ＝ ATR × この値 |
| `tp_atr_multiplier` | 1.5 | 利確幅 ＝ ATR × この値 |
| `breakeven_trigger_r` | 1.0 | 建値ストップの発動点（R）。`None` で無効（fixedSL 系のみ） |
| `price_decimals` | 5 | 価格の丸め桁数（クロス円は3） |
| `start_date` / `end_date` | None | 検証期間（`"YYYY-MM-DD"`）。None なら全期間 |
| `execution_timeframe` | `"1min"` | `"1min"` / `"signal"`（D1版は `"daily"`） |
| `chart_timeframe` | `"signal"` | チャートの表示単位（表示だけで、結果には影響しない）。`"signal"` / `"native"` |
| `swing_point_style` | `"dash"` | 判定足実行時のスイングの表示。`"dash"` / `"circle"` |
| `margin` | 1/25 | 証拠金率（レバレッジ25倍） |
| `csv_filename` / `html_filename` | `trade_history.csv` / `swing_breakout_chart.html` | 単発検証の出力ファイル名 |
| `grid_enabled` | False | SL×TP グリッド検証（4H / fixedSL のみ） |
| `grid_sl_values` / `grid_tp_values` | (0.8, 1.0, 1.2, 1.5) / (1.0, 1.5, 2.0, 2.5) | グリッドの候補 |
| `grid_csv_filename` | `sl_tp_grid.csv` | グリッドの一覧表 |
| `grid_save_trades` / `grid_trades_dir` | True / `grid_trades` | 組み合わせごとの取引履歴の保存 |

## 6. ルートにある過去の出力ファイル

| ファイル | 内容 |
|---|---|
| `trade_history.csv` / `swing_breakout_chart.html` | 単発検証の最新の出力（実行のたびに上書き） |
| `trade_history_EURUSD_20210101-20251231_4H_sl1.5_tp2.5.csv` / 同名 `.html` | EURUSD 4時間足 SL1.5/TP2.5 の結果 |
| `trade_history_..._slFixed.csv` | 同じ条件の損失額固定版（fixedSL）の結果 |
| `sl_tp_grid.csv` / `grid_trades/` | SL×TP グリッド検証の結果 |
| `diagnosis/` | `diagnose_trades.py` の出力 |

## 7. 実行時の注意
- 実行はリポジトリのルートで行う（`histData/` などを相対パスで探すため）。
- 必要なライブラリ: `backtesting`、`pandas`、`numpy`、`bokeh`。Bokeh 3.x では取引の矢印マークが表示されないという警告が出る（スクリプトは Bokeh 2.4.3 への変更を案内している）。
- 1分足で5年分を実行すると時間がかかる。グリッド検証や新しいルールは、短い期間で所要時間を確かめてから広げる。
- 各スクリプトは最後にチャートをブラウザで開く。ブラウザの無い環境（クラウドなど）では何もせずに終わる。
