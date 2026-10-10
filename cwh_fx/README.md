# カップウィズハンドル FX（ATRベース）

`cup_with_handle_fx_spec.md`（仕様書）の実装。FX ペアのみ。データは既存の `main_4H_fixedSL.load_gmo_click_1min_data()` で
`histData/<ペア>/*.zip`（GMOクリック証券の1分足）を読み、NY 17:00 区切りの足（既定は日足、BID）にまとめる。

## 実行（リポジトリのルートで）
```
python cwh_fx/run_backtest.py                                    # config.yaml の設定（仕様の初期値）
python cwh_fx/run_backtest.py --set "bar_hours=4" --label h4      # 一時的に上書き（config.yaml は書き換えない）
python cwh_fx/run_backtest.py \
  --variant spec "A 仕様どおり（日足）" "" \
  --variant d1_nocontr "B 日足・ATR収縮なし" "handle_atr_contraction=99" \
  --variant h4_nocontr "C 4時間足・ATR収縮なし" "bar_hours=4 handle_atr_contraction=99" \
  --validate h4_nocontr                                          # 並べて比較し、C に仕様12章の検証をかける（約7分）
python -m pytest cwh_fx/tests                                     # テスト（合成データ・先読み・決済ルール）
python cwh_fx/report_html.py cwh_results/<実行フォルダ>             # レポートだけ作り直す（findings.json があればその文章を使う）
```
1分足の読み込みは1ペア10〜15秒かかるので、`cache/`（.gitignore 済み）に1分足と足を pickle で保存する。

## ファイル
| ファイル | 内容 |
|---|---|
| `config.yaml` | パラメータ（仕様2の初期値）、資金管理、コスト（スプレッドはデータの平均 ASK−BID を切り上げた値） |
| `src/data.py` | GMO 1分足 → 足（日足のラベルは取引日）。キャッシュ |
| `src/indicators.py` | TR・ATR（ワイルダー）・SMA・スイング |
| `src/pattern.py` | `detect_cup_with_handle(df, params, t, direction)`。ショートは価格を反転して同じ関数 |
| `src/signals.py` | セットアップの管理（失効・破棄）とブレイクのシグナル |
| `src/backtest.py` | イベント駆動のポートフォリオ・バックテスト（同時保有・通貨エクスポージャー・優先順位） |
| `src/sizing.py` / `src/costs.py` | ロット計算・口座通貨換算（USD 経由） / スプレッド・スリッページ・スワップ |
| `src/report.py` / `src/diagnostics.py` / `src/validation.py` | 成績指標 / どの条件で落ちたか / 仕様12.3〜12.7 の検証 |
| `src/charts.py` | 取引を `dow_swing_chart.make_chart()` 用の形に直す |
| `run_backtest.py` / `report_html.py` | 実行・保存 / レポート（`cwh_results/<日時>_<ラベル>/report.html`） |

## 仕様に書かれていない部分の決め方
- 右縁は仕様4.3の文面どおり `H_right >= H_left − 1×ATR50` の下限だけ（`right_rim_mode: range` で上下とも）。
- 左縁からカップの底までに左縁より高い足が無いこと（`left_rim_highest`）。左縁の候補が複数成立したら、いちばん新しいもの。
- ATR50 は左縁・カップの値に `ATR50_{i_left}`、ハンドルの値に `ATR50_t`。
- 「ハンドル完成」= 初めて全条件を満たした足。そこから10本でブレイクしなければ失効。その時点以降のハンドル安値を終値で割ったら破棄。
- 損切り・部分利確は足の高値・安値で判定し、同じ足で両方に届いたら損切りだけ。部分利確後の建値ストップは次の足から。
- チャンデリアのラインは有利な方向にだけ動かす（`trail_ratchet`）。タイムストップは40本目の終値で判定。
- 同じペアは1ポジションまで（`one_position_per_pair`）。エントリー時の資産は前の足の終値時点の時価。
- 買いは ASK（BID + スプレッド）で約定し BID で決済、売りはその逆。スワップは表が空なら 0（警告を出す）。

## これまでの実行
| フォルダ | 内容 |
|---|---|
| `cwh_results/20261010_1213_cwh_spec_vs_relaxed` | 仕様の初期値（日足）は 2020〜2025・5ペアで0件。ATR収縮を外すと日足5件 −2.2R、4時間足20件 +2.7R（ランダム比較 p≈0.24） |
