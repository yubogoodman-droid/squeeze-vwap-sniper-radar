# 幣安 USDT-M 永續合約「Squeeze × VWAP 護盤 × 主動買盤」雙模突破掃描器與誠實回測看板

100% 獨立運作的純訊號偵測與歷史回測視覺化工具（**無任何下單功能、不需 API Key**，全數使用幣安官方 `https://fapi.binance.com` 公開行情端點）。

## 核心防偏誤與誠實回測設計

1. **即時掃描預設剔除未收盤 K 線（消除盤中重繪 Repainting）**：
   - `squeeze_breakout_radar.py` 與單一前端來源 `index.html`（`radar.html` 自動導向 `index.html`）預設皆為 `drop_forming=True`，僅對**已完整收盤**之 K 線計算指標與觸發訊號；如需盤中預警可透過 `--include-forming`（網頁勾選框或 `?include_forming=1`）顯式開啟。
   - `--watch` 輪詢模式支援 `--seen-file` 原子 JSON 持久化（瀏覽器端同步寫入 `localStorage`），重啟不會重複推播舊訊號。
2. **無偏誤動態歷史標的池（消除倖存者、強制塞幣與字母序截斷偏誤）**：
   - 移除硬編碼特定幣種與快取字母序截斷邏輯；支援由 `--cache-dir` / `--npz-dir`（或環境變數 `BINANCE_CACHE_DIR` / `BINANCE_NPZ_DIR`）載入全數合約快取（相容 `<SYMBOL>_15m_365d.pkl` 格式），或透過 `/fapi/v1/exchangeInfo` 納入回測期間曾存續之已下架永續合約（`list_historical_usdt_perps`）。
   - 每筆訊號觸發前嚴格以 **`[t - 24h, t)` 過去 24 小時滾動成交額**（不含當根突破量與未來成交量）進行時點流動性過濾與排序（`PointInTimeVolumeIndex`）。
3. **去過擬合、參數透明化與樣本外盲測（IS / OOS + Walk-Forward）**：
   - 移除 `dist_s60` `2.0%~3.7%` 事後挖洞過濾條件，並將核心突破與風控門檻完整公開為具預設值說明之 CLI 參數。
   - 內建參數凍結之樣本內（In-Sample）vs 樣本外（Out-of-Sample，`--oos-date` / `--oos-ratio`）對照表與時間序列滾動切片（`--wf-folds`）。
4. **保守盤中出場與跳空成交模擬**：
   - 浮盈達 `+0.55R` 上移保本止損之同一根 K 線，立即保守檢查當根 `Low` 是否跌破新止損；跳空穿過止損時按 `min(Open, Stop)` 真實跳空價成交；收盤跌破 `BB` 中軌時按實際 `Close` 出場（不以止損價墊高收盤崩跌）。
5. **完整交易成本與費滑敏感度分析**：
   - 納入 `/fapi/v1/fundingRate` 歷史資金費率（離線時自動退回 `--funding-rate-8h 0.03`，即每跨 8h UTC 結算扣 `0.03%`）。
   - 雙邊手續費+滑價可由 `--fee-pct` 設定（預設 `0.12%`），並於終端機與 HTML 報表同步輸出 **`0.12%` / `0.30%` / `0.50%`** 三段費滑敏感度對照。
6. **誠實資金模型與盤中 Mark-to-Market 最大回撤（MTM MDD）**：
   - **主指標（Headline）**：支援多倉併發上限（`--max-positions 3`）之 **2% 固定風險比例複利（Fixed-Fractional Risk）**。
   - **1x 真實無槓桿複利** vs 明確標示之「單筆報酬算術總和（非複利）」。
   - **3x 壓力測試對照**：納入分級維持保證金強平（`check_long_liquidation`）與單筆流動性上限（預設突破 K 線成交額 `2%` / 前 24h 成交額 `0.5%`）。
   - 所有資金曲線回撤均按持倉期間每根 K 線盤中最低價逐根計算 **Mark-to-Market Intrabar MDD**；獲利因子（PF）於零虧損或零交易時誠實顯示 `inf` / `N/A`（絕不硬編碼 `999`）。
7. **跨週期視窗自動縮放與連線韌性**：
   - `5m` / `15m` / `30m` / `1h` 各項指標與形態視窗皆依 K 線毫秒時長自動換算物理時間長度，不再寫死 `15m` 根數。
   - 內建 `418` / `429` 限流退避重試與日誌告警；網路中斷或抓取引發失敗時絕不寫入殘缺快取檔案。

## 快速使用

### 1. 瀏覽器即時掃描
直接開啟 `index.html`（`radar.html` 會自動導向同一來源），純前端連線 `https://fapi.binance.com` 公開行情，預設僅掃描已收盤 K 線。

### 2. Python CLI 即時掃描與盯盤

```bash
# 單次全市場掃描（僅掃描已收盤 K 線）
python3 squeeze_breakout_radar.py --scan --interval 15m --plot

# 60 秒輪詢盯盤（已推播訊號自動持久化至磁碟）
python3 squeeze_breakout_radar.py --watch 60 --interval 15m --seen-file breakout_output/.watch_seen_squeeze.json
```

### 3. 離線快取 / 全市場誠實回測（相容 `<SYMBOL>_15m_365d.pkl`）

```bash
# 從本地快取目錄離線重跑 365 天完整回測（含 IS/OOS、Walk-Forward 與 0.12%/0.30%/0.50% 費滑敏感度）
python3 backtest_180d_squeeze.py --days 365 --cache-dir breakout_output/cache_365d --min-vol 2000000 --oos-ratio 0.30 --wf-folds 4 --no-fetch-funding

# 180 天回測（若環境可直連 fapi.binance.com 可省略 --no-fetch-funding 以抓取歷史真實資金費率）
python3 backtest_180d_squeeze.py --days 180 --cache-dir breakout_output/cache_180d --min-vol 2000000 --oos-ratio 0.30 --wf-folds 4
```

> **靜態圖卡說明**：`breakout_output/` 目錄內既有提交之靜態 HTML/PNG 為舊版快照；於備有 `<SYMBOL>_15m_365d.pkl` 快取或可連線 `fapi.binance.com` 之環境執行上述指令，即可覆寫產出最新誠實回測看板。

### 4. 執行單元測試

```bash
python3 -m pytest -v tests/
```
