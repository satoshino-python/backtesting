# CLAUDE.md

FX（GMOクリック証券の1分足）のスイングブレイクアウト戦略を backtesting.py で検証するリポジトリ。
各スクリプトのルール・設定・出力は [docs/STRATEGIES.md](docs/STRATEGIES.md)、ダウ理論判定の仕様は
[docs/DOW_TREND_SPEC.md](docs/DOW_TREND_SPEC.md) にまとめてある。作業の前に読むこと。

## 実行
- リポジトリのルートで実行する（`histData/` などを相対パスで探す）。ローカルの Python は `.venv/`。
- Windows では出力の絵文字で落ちるため `PYTHONIOENCODING=utf-8` を付ける。
- 1分足で5年分は時間がかかる（1回あたり数分）。新しいルールは短い期間で確かめてから広げる。
- `main_4H_fixedSL.py` は他のスクリプトから import されている。変更すると multi / dow 版の結果も変わる。
- `fast_engine.py` は `SwingBreakoutStrategy1Min`（固定SL/TP）と `SwingBreakoutTrail1Min`（1時間足トレーリング）の売買ルールを
  backtesting.py なしで再現した高速版（約100倍速い）。複数ペアは `main_4H_fixedSL_fast_multi.py` で並列に検証できる。
  戦略のルールを変えたら `fast_engine.py` も合わせて直し、`compare_fast_engine.py` で backtesting.py との一致を確かめる。

## 検証結果のチャート（毎回作る）
トレードを伴う検証では、backtesting.py 標準のチャートに加えて、**トレード付きのダウ理論チャート**を必ず出力する。
`dow_swing_chart.make_chart()` を呼ぶだけでよい（`main_4H_fixedSL_dow.py` は `TRADE_CHART = True` で自動的に出力する）。

```python
from dow_swing_chart import make_chart

make_chart(
    df_1min,                      # load_gmo_click_1min_data() の1分足。助走期間を含む全期間
    out_dir / f"dow_trade_chart_{symbol}_{period}_{mode}.html",
    f"{symbol} {start} ~ {end} / トレード: {mode}",
    start, end,
    trades=stats["_trades"],      # backtesting.py の取引履歴（そのCSVのパスでも可）
    weekly_trend=used_weekly,     # 各週に使った週足トレンド（前週末の判定）。使わない戦略は None
    mode=mode, risk=cfg.cash * cfg.risk_pct,
    entry_window=cfg.window, entry_atr=cfg.atr_period, price_decimals=cfg.price_decimals,
)
```

- 表示内容: エントリー（▲買い / ▼売り）・決済（●）・SL/TP、背景の週足トレンド、4時間足のエントリーライン、前/次のトレード移動。
- 方向フィルターを使った検証では `filter_state=`（1分足ごとの 1=買いだけ許可 / -1=売りだけ許可 / 0=見送り / 2=両方）と
  `filter_label=` を渡す。背景が「検証で使ったフィルター」になり、トレードの向きと背景が一致しているかを確かめられる（例: `filter_report_charts.py`）。
- 押し・戻りの回数を条件にした検証では `waves=` に「どの波を数えたか」を渡す（`pullback_count.pullback_waves()` の結果。起点の日足スイング◆と、数えた4時間足のスイング①②③を表示。`dow_d1_h4_breakout.py` が使用）。
- 移動平均（表示中の時間足の SMA、既定 20/50/120）を表示する。期間は `ma_periods=` とチャートの設定で変えられる。
- 既存の取引履歴CSVからは `python dow_swing_chart.py --trades dow_results/trades_<期間>_<モード>.csv` で作れる。
- チャートのテンプレートは `dow_swing_chart_template.html`。JavaScript の `computeDow` は `dow_trend.py` の移植なので、
  片方を変えたらもう片方も合わせる。

## 結果とチャートはアーティファクトで見せる（毎回）
検証を実行したら、結果とチャートを必ず claude.ai のアーティファクト（Artifact ツール）として公開し、そのページで報告する。
CSV や HTML をリポジトリに保存するだけ、またはチャットに表を貼るだけで終わらせない。

- 1回の実行につき1つのアーティファクト。中身は次の順に並べる。
  1. 主な数字（合計R・PF・最大DD・取引数。比較する実行があれば、その値と並べる）
  2. 損益曲線、年別・ペア別の合計R、成績一覧
  3. トレード付きのダウ理論チャート（ペアごと）。`make_chart()` で作った HTML を `files` で同じアーティファクトに載せ、一覧ページからリンクする
  4. 売買ルール・設定と、結果を読むときの注意点（期間分割の有無、合わせ込みの可能性など）
- スマホで見る前提で作る（1カラム、表は横スクロール、ライト/ダークの両方で読める）。
  既存の例: 「スイングブレイクアウト高速版 6ペア」「トレンドフィルター検証 6ペア」（`filter_results/trend_filter_report.html`）と同じ見た目に揃える。
- 同じ検証を設定を変えて実行し直したときは、前のアーティファクトを上書きせず別のアーティファクトにする（下の「結果の保存」と同じ考え方）。
  同じ実行の修正（表の誤り、説明の追加など）は同じアーティファクトを更新する。
- 1ページ・1ファイルあたり 16MB まで。ダウ理論チャートは1ペア約 2.7MB なので、ページ本体に埋め込まず別ファイルにする。
- 公開したページの HTML はリポジトリにも結果フォルダ（例: `fast_results/<日時>_<ラベル>/report.html`）に保存する。
- Artifact ツールが使えない環境（ローカルのターミナルなど）では、同じ内容の HTML を結果フォルダに保存し、そのパスを伝える。

## 結果の保存
- 実行ごとに結果を別フォルダ・別ファイル名に保存し、前の結果を上書きしない（`multi_results/<日時>_<ラベル>/` など）。
- 設定を一時的に変えて実行するときは、スクリプトの `CFG` を書き換えずに `dataclasses.replace` で上書きして呼び出す。
