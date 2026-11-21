from __future__ import annotations

import ast
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, List
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import backtest_180d_squeeze as bt_sq
import backtest_utils as bt_utils
import squeeze_breakout_radar as sq_radar

TPE = ZoneInfo("Asia/Taipei")


def _make_raw_klines(n: int = 275, bar_ms: int = 900_000, include_forming: bool = True) -> List[List[Any]]:
    now_ms = int(time.time() * 1000)
    current_bar_open = (now_ms // bar_ms) * bar_ms
    last_open = current_bar_open if include_forming else (current_bar_open - bar_ms)
    start_open = last_open - (n - 1) * bar_ms

    rows: List[List[Any]] = []
    for idx in range(n):
        t_open = start_open + idx * bar_ms
        # Tight coil on closed bars; massive climax spike ONLY on the unclosed forming bar
        if include_forming and idx == n - 1:
            o, h, l, c = 100.0, 106.0, 99.95, 105.8
            vol, qv, tbv = 60_000.0, 6_200_000.0, 45_000.0
        else:
            drift = 100.0 + 0.002 * idx
            o = drift
            h = drift * 1.0015
            l = drift * 0.9985
            c = drift * 1.0004
            vol = 1_000.0
            qv = 100_000.0
            tbv = 530.0
        rows.append(
            [
                t_open,
                f"{o:.6f}",
                f"{h:.6f}",
                f"{l:.6f}",
                f"{c:.6f}",
                f"{vol:.4f}",
                t_open + bar_ms - 1,
                f"{qv:.4f}",
                100,
                f"{tbv:.4f}",
                f"{tbv * c:.4f}",
                "0",
            ]
        )
    return rows


def test_signal_only_fapi_and_hygiene() -> None:
    assert sq_radar.BINANCE_FAPI == "https://fapi.binance.com"
    assert bt_utils.BINANCE_FAPI == "https://fapi.binance.com"

    py_files = [
        REPO_ROOT / "squeeze_breakout_radar.py",
        REPO_ROOT / "backtest_180d_squeeze.py",
        REPO_ROOT / "backtest_utils.py",
    ]
    for fpath in py_files:
        src = fpath.read_text(encoding="utf-8")
        assert "/fapi/v1/order" not in src
        assert "X-MBX-APIKEY" not in src
        assert "EVAAUSDT" not in src
        assert "C:\\Users" not in src
        assert "C:\\tmp" not in src
        assert "999.0" not in src

        # Ensure zero silent `except ...: pass` or bare `except:` blocks
        tree = ast.parse(src, filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                assert node.type is not None, f"Bare except found in {fpath.name}"
                assert not (
                    len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                ), f"Silent except-pass found in {fpath.name}"

    sq_src = (REPO_ROOT / "backtest_180d_squeeze.py").read_text(encoding="utf-8")
    radar_src = (REPO_ROOT / "squeeze_breakout_radar.py").read_text(encoding="utf-8")
    index_html = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
    radar_html = (REPO_ROOT / "radar.html").read_text(encoding="utf-8")
    readme_src = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "max(closes" not in sq_src
    assert "+0.6R" not in sq_src
    assert "dist_s60" not in radar_src
    assert "distS60" not in index_html
    assert "90.5%" not in index_html
    assert "90.5%" not in radar_html
    assert "90.5%" not in readme_src
    assert "999.00" not in readme_src

    # Verify index.html vs radar.html deduplication (one source of truth)
    assert "function analyzeSymbol" in index_html
    assert "function analyzeSymbol" not in radar_html
    assert "index.html" in radar_html
    assert "dropForming = true" in index_html
    assert "localStorage" in index_html


def test_live_scanner_drops_forming_candle_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _make_raw_klines(n=275, bar_ms=900_000, include_forming=True)
    forming_open_ms = int(raw[-1][0])

    monkeypatch.setattr(sq_radar, "api_get", lambda path, params=None, retries=4: list(raw))

    df_closed = sq_radar.load_klines_with_orderflow("TESTUSDT", interval="15m", limit=270)
    assert not df_closed.empty
    assert bt_utils.ts_to_ms(df_closed.index[-1]) < forming_open_ms
    hits_closed = sq_radar.scan_squeeze_breakouts(df_closed, "TESTUSDT", interval="15m")
    assert hits_closed == []

    df_forming = sq_radar.load_klines_with_orderflow("TESTUSDT", interval="15m", limit=270, drop_forming=False)
    assert bt_utils.ts_to_ms(df_forming.index[-1]) == forming_open_ms
    hits_forming = sq_radar.scan_squeeze_breakouts(df_forming, "TESTUSDT", interval="15m")
    assert len(hits_forming) == 1
    assert bt_utils.ts_to_ms(hits_forming[0].timestamp) == forming_open_ms


def test_pit_volume_ranking_historical_perps_and_365d_cache(tmp_path: Path) -> None:
    idx = pd.date_range("2026-06-01", periods=200, freq="15min", tz=TPE)
    # EARLYUSDT has real prior liquidity; SPIKEUSDT is illiquid before bar 100 and spikes ONLY ON bar 100 + future
    early_qv = np.where(np.arange(200) < 110, 100_000.0, 1_000.0)
    spike_qv = np.where(np.arange(200) < 100, 1_000.0, 5_000_000.0)

    dfs = {
        "EARLYUSDT": pd.DataFrame({"Close": 1.0, "Volume": early_qv, "QuoteVol": early_qv}, index=idx),
        "SPIKEUSDT": pd.DataFrame({"Close": 1.0, "Volume": spike_qv, "QuoteVol": spike_qv}, index=idx),
    }
    pit = bt_utils.PointInTimeVolumeIndex(dfs)

    # Strictly before bar 100 ([t - 24h, t)), SPIKEUSDT's bar-100 spike must NOT be counted!
    ok_early, qv_e, rank_e = pit.check_eligibility("EARLYUSDT", idx[100], min_quote_vol=1_000_000.0, top_n=1)
    ok_spike, qv_s, rank_s = pit.check_eligibility("SPIKEUSDT", idx[100], min_quote_vol=1_000_000.0, top_n=1)
    assert ok_early and rank_e == 1 and qv_e == pytest.approx(9_600_000.0)
    assert not ok_spike and rank_s == 2 and qv_s == pytest.approx(96_000.0)

    # Delisted/settling perpetuals active during backtest window must be included
    fake_session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.raise_for_status.return_value = None
    w_start = bt_utils.ts_to_ms(idx[0])
    w_end = bt_utils.ts_to_ms(idx[-1])
    resp.json.return_value = {
        "symbols": [
            {
                "symbol": "DELISTEDUSDT",
                "quoteAsset": "USDT",
                "contractType": "PERPETUAL",
                "status": "SETTLING",
                "onboardDate": w_start - 86_400_000 * 30,
                "deliveryDate": w_start + 86_400_000,
            },
            {
                "symbol": "PREWINDOWUSDT",
                "quoteAsset": "USDT",
                "contractType": "PERPETUAL",
                "status": "CLOSE",
                "onboardDate": w_start - 86_400_000 * 90,
                "deliveryDate": w_start - 86_400_000,
            },
            {
                "symbol": "FUTURECOINUSDT",
                "quoteAsset": "USDT",
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "onboardDate": w_end + 86_400_000 * 10,
                "deliveryDate": 4133404800000,
            },
        ]
    }
    fake_session.get.return_value = resp
    hist_syms = bt_utils.list_historical_usdt_perps(fake_session, w_start, w_end)
    assert "DELISTEDUSDT" in hist_syms
    assert "PREWINDOWUSDT" not in hist_syms
    assert "FUTURECOINUSDT" not in hist_syms

    # Verify sibling repo cache format (<SYMBOL>_15m_365d.pkl with QuoteVolume) + NPZ loading without alpha slicing
    cdir = tmp_path / "cache365"
    ndir = tmp_path / "npz"
    cdir.mkdir()
    ndir.mkdir()
    for sym in ("AAAUSDT", "MMMUDST_BAD", "ZZZUSDT"):
        if not sym.endswith("USDT"):
            continue
        pd.DataFrame(
            {
                "Open": 10.0,
                "High": 10.1,
                "Low": 9.9,
                "Close": 10.0,
                "Volume": 100.0,
                "QuoteVolume": 1000.0,
                "TakerBuyVol": 55.0,
            },
            index=idx[:50],
        ).to_pickle(cdir / f"{sym}_15m_365d.pkl")

    ts_arr = np.array([bt_utils.ts_to_ms(t) for t in idx[:50]], dtype=np.int64)
    np.savez(
        ndir / "NPZCOINUSDT_15m.npz",
        ts=ts_arr,
        o=np.full(50, 5.0),
        h=np.full(50, 5.1),
        l=np.full(50, 4.9),
        c=np.full(50, 5.0),
        v=np.full(50, 200.0),
        qv=np.full(50, 1000.0),
        tbv=np.full(50, 110.0),
    )
    loaded = bt_utils.load_cached_frames([cdir, ndir], interval="15m")
    assert set(loaded.keys()) == {"AAAUSDT", "ZZZUSDT", "NPZCOINUSDT"}
    assert "QuoteVol" in loaded["ZZZUSDT"].columns


def test_m1_former_dist_s60_gap_allowed_and_cli_thresholds() -> None:
    """Verify M1 signals with dist_s60 inside the former 2.0%-3.7% excluded hole are now accepted."""
    n = 320
    idx = pd.date_range("2026-07-01", periods=n, freq="15min", tz=TPE)
    c = np.empty(n, dtype=float)
    for k in range(n):
        if k < 250:
            c[k] = 97.5
        elif k < 255:
            c[k] = 101.0 + (k - 250) * 0.9
        elif k < 300:
            base = 105.2 + (k - 255) * 0.042
            osc = 0.52 * np.sin(k * 1.1) if k < 292 else 0.12 + 0.02 * (k - 292)
            c[k] = base + osc
        else:
            c[k] = 111.5

    o = c * 0.999
    h = c * 1.0015
    l = c * 0.9985
    v = np.full(n, 1_000.0)
    qv = v * c
    tbv = np.full(n, 520.0)

    # Ensure bars 292..299 (8 bars prior to 300) stay above 12h VWAP with healthy TBR
    for k in range(292, 300):
        c[k] = 107.05 + 0.03 * (k - 292)
        o[k] = c[k] - 0.04
        h[k] = c[k] + 0.06
        l[k] = c[k] - 0.08
        v[k] = 1_000.0
        qv[k] = v[k] * c[k]
        tbv[k] = 520.0

    # Breakout bar 300: > Donchian24h * 1.004, ~10x volume (NOT >=30x so it can ONLY be M1, not M2!), CLV >= 0.72
    o[300] = 107.30
    l[300] = 107.25
    h[300] = 111.80
    c[300] = 111.50
    v[300] = 10_000.0
    qv[300] = v[300] * c[300]
    tbv[300] = 6_000.0

    df = sq_radar.compute_radar_indicators(
        pd.DataFrame(
            {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v, "QuoteVol": qv, "TakerBuyVol": tbv},
            index=idx,
        ),
        interval="15m",
    )
    dist_s60_at_299 = (df["Close"].iloc[299] - df["SMA60"].iloc[299]) / df["SMA60"].iloc[299] * 100.0
    assert 2.0 < dist_s60_at_299 < 3.7, f"Expected dist_s60 in (2.0, 3.7), got {dist_s60_at_299:.2f}%"

    hits = sq_radar.scan_squeeze_breakouts(df, "TESTM1USDT", interval="15m")
    m1_hits = [x for x in hits if x.bar_idx == 300 and x.mode == "VWAP_COIL_BREAK"]
    assert len(m1_hits) == 1

    # Custom CLI threshold config filters it out when qv_mult_min is raised above 10x
    strict_cfg = sq_radar.RadarConfig(qv_mult_min=20.0)
    assert sq_radar.scan_squeeze_breakouts(df, "TESTM1USDT", interval="15m", config=strict_cfg) == []


def test_exits_intrabar_raise_gap_through_and_no_crash_clamp() -> None:
    idx = pd.date_range("2026-08-01", periods=20, freq="15min", tz=TPE)
    base_hit = sq_radar.RadarHit(
        symbol="TESTUSDT", interval="15m", mode="VWAP_COIL_BREAK", bar_idx=5,
        timestamp=idx[5], price=100.0, stop_price=95.0, breakout_level=99.5, vwap=99.0,
        quote_vol_usd=500_000.0, qv_mult=8.0, vol_accel=2.0, taker_buy_pct=60.0,
        bar_gain_pct=3.0, clv=0.8, bbw_min10_pct=2.0, bbw_max24_pct=4.0,
        rng24_pct=3.0, rng96_pct=5.0, score=85, max_gain_after_pct=30.0, note="test",
    )

    # 1) Intrabar stop raise check on bar 6: High arms breakeven (100.3), Low (99.5) pierces 100.3
    opens = np.full(20, 100.0)
    highs = np.full(20, 101.0)
    lows = np.full(20, 99.6)
    closes = np.full(20, 100.5)
    highs[6] = 107.0
    lows[6] = 99.5
    closes[6] = 102.0
    for k in range(7, 20):
        opens[k], highs[k], lows[k], closes[k] = 120.0, 132.0, 119.0, 130.0
    df_raise = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "BB_MID": np.full(20, 98.0)},
        index=idx,
    )
    tr1 = bt_sq.simulate_trade(df_raise, base_hit, roundtrip_cost_pct=0.12)
    assert tr1 is not None
    assert tr1.exit_idx_trend == 6
    assert tr1.exit_reason_trend == "保本止盈"
    assert tr1.exit_price_trend == pytest.approx(100.3)

    # 2) Gap-through open below stop: stop is 95.0, bar 6 opens at 88.0 -> fills at min(open, stop) = 88.0!
    opens_gap = np.full(20, 100.0)
    highs_gap = np.full(20, 101.0)
    lows_gap = np.full(20, 99.0)
    closes_gap = np.full(20, 100.0)
    opens_gap[6], highs_gap[6], lows_gap[6], closes_gap[6] = 88.0, 89.0, 85.0, 86.0
    df_gap = pd.DataFrame(
        {"Open": opens_gap, "High": highs_gap, "Low": lows_gap, "Close": closes_gap, "BB_MID": np.full(20, 99.0)},
        index=idx,
    )
    tr2 = bt_sq.simulate_trade(df_gap, base_hit, roundtrip_cost_pct=0.12)
    assert tr2 is not None
    assert tr2.exit_idx_trend == 6
    assert tr2.exit_reason_trend == "初始止損"
    assert tr2.exit_price_trend == pytest.approx(88.0)
    assert tr2.pnl_fixed3r == pytest.approx(-12.0 - 0.12)

    # 3) BB_MID close exit uses actual closes[k] without clamping to cur_stop
    opens_bb = np.full(20, 103.0)
    highs_bb = np.full(20, 104.0)
    lows_bb = np.full(20, 101.0)
    closes_bb = np.full(20, 103.5)
    bb_mid = np.full(20, 102.0)
    # Bar 6 arms breakeven (100.3), stays above stop; Bar 8 closes at 101.2 (< BB_MID 102.0, > stop 100.3)
    opens_bb[8] = 102.4
    highs_bb[8] = 102.5
    lows_bb[8] = 101.0
    closes_bb[8] = 101.2
    df_bb = pd.DataFrame(
        {"Open": opens_bb, "High": highs_bb, "Low": lows_bb, "Close": closes_bb, "BB_MID": bb_mid},
        index=idx,
    )
    tr3 = bt_sq.simulate_trade(df_bb, base_hit, roundtrip_cost_pct=0.12)
    assert tr3 is not None
    assert tr3.exit_idx_trend == 8
    assert tr3.exit_reason_trend == "跌破BB中軌止盈"
    assert tr3.exit_price_trend == pytest.approx(101.2)


def test_funding_rate_store_configurable_fees_and_sensitivity(tmp_path: Path) -> None:
    t0 = pd.Timestamp("2026-07-01 07:45:00", tz="UTC")
    t1 = pd.Timestamp("2026-07-01 16:00:00", tz="UTC")
    store_offline = bt_utils.FundingRateStore(cache_dir=tmp_path, fallback_rate_8h_pct=0.03, fetch_online=False)
    cost_est, n_ev, src = store_offline.compute_funding_cost_pct(
        "BTCUSDT", bt_utils.ts_to_ms(t0), bt_utils.ts_to_ms(t1), bar_ms=900_000
    )
    assert src == "est" and n_ev == 1 and cost_est == pytest.approx(0.03)

    fdir = tmp_path / "funding"
    fdir.mkdir(parents=True, exist_ok=True)
    ts_1600 = int(pd.Timestamp("2026-07-01 16:00:00", tz="UTC").timestamp() * 1000)
    pd.DataFrame(
        {"fundingTime": [ts_1600 - 28_800_000, ts_1600, ts_1600 + 28_800_000], "fundingRate": [0.0001, 0.0006, 0.0002]}
    ).to_pickle(fdir / "ETHUSDT_funding.pkl")
    store_cached = bt_utils.FundingRateStore(cache_dir=tmp_path, fallback_rate_8h_pct=0.03, fetch_online=False)
    cost_fapi, n_fapi, src_fapi = store_cached.compute_funding_cost_pct(
        "ETHUSDT", bt_utils.ts_to_ms(t0), bt_utils.ts_to_ms(t1), bar_ms=900_000
    )
    assert src_fapi == "fapi" and n_fapi == 1 and cost_fapi == pytest.approx(0.06)

    # Sensitivity at 0.12% / 0.30% / 0.50% flips a +0.30% breakeven exit from a tiny win to a loss
    dt0 = datetime(2026, 7, 1, 12, 0, tzinfo=TPE)
    be_trade = bt_utils.PortfolioTradeInput(
        symbol="BEUSDT",
        entry_time=dt0,
        exit_time=dt0 + timedelta(hours=1),
        entry_price=100.0,
        stop_price=95.0,
        exit_price=100.30,
        risk_pct=5.0,
        net_pnl_pct=0.30 - 0.12,
        r_mult=(0.30 - 0.12) / 5.0,
        worst_low_price=99.8,
        roundtrip_cost_pct=0.12,
        bar_quote_vol_usd=5_000_000.0,
        trailing_24h_qv_usd=50_000_000.0,
    )
    sens = bt_utils.compute_cost_sensitivity([be_trade], cost_levels_pct=(0.12, 0.30, 0.50))
    assert [r["cost_pct"] for r in sens] == [0.12, 0.30, 0.50]
    assert sens[0]["stats"]["wins"] == 1 and sens[0]["stats"]["sum_r"] > 0
    assert sens[1]["stats"]["wins"] == 0 and sens[1]["stats"]["sum_r"] == pytest.approx(0.0)
    assert sens[2]["stats"]["wins"] == 0 and sens[2]["stats"]["sum_r"] < 0


def test_money_model_mtm_intrabar_mdd_liquidation_and_pf() -> None:
    # 1) Honest PF formatting (no 999)
    assert bt_sq.summarize_trades([])["pf"] is None
    assert bt_utils.fmt_pf(None) == "N/A"
    assert bt_utils.fmt_pf(float("nan")) == "N/A"
    assert bt_utils.fmt_pf(float("inf")) == "inf"
    assert bt_utils.compute_profit_factor([5.0, 3.0]) == float("inf")
    assert bt_utils.fmt_pf(bt_utils.compute_profit_factor([6.0, -2.0])) == "3.00"

    # 2) Mark-to-market intrabar MDD must be > 0 when a winning trade dips -8% intrabar before rallying!
    t0 = datetime(2026, 5, 1, 0, 0, tzinfo=TPE)
    bar_tms = np.array(
        [bt_utils.ts_to_ms(t0 + timedelta(minutes=15 * k)) for k in (1, 2, 3)],
        dtype=np.int64,
    )
    dip_then_win = bt_utils.PortfolioTradeInput(
        symbol="DIPWINUSDT",
        entry_time=t0,
        exit_time=t0 + timedelta(minutes=45),
        entry_price=100.0,
        stop_price=90.0,
        exit_price=120.0,
        risk_pct=10.0,
        net_pnl_pct=19.88,
        r_mult=1.988,
        worst_low_price=92.0,
        bar_quote_vol_usd=20_000_000.0,
        trailing_24h_qv_usd=200_000_000.0,
        mtm_times_ms=bar_tms,
        mtm_trough_prices=np.array([92.0, 105.0, 118.0]),
        mtm_close_prices=np.array([95.0, 125.0, 120.0]),
    )
    port_mtm = bt_utils.simulate_honest_portfolio([dip_then_win], initial_cap=1000.0, max_positions=1)
    assert port_mtm["spot_1x"]["ret_pct"] > 19.0
    assert port_mtm["spot_1x"]["mdd_pct"] > 7.5  # Captures the -8% intrabar dip to 92.0!
    assert port_mtm["headline_risk"]["mdd_pct"] > 1.5  # 20% notional * -8% dip = ~1.6% equity dip!

    # 3) Liquidity cap & tiered maintenance-margin liquidation on 3x stress test vs survival of 2% risk
    tr_micro = bt_utils.PortfolioTradeInput(
        symbol="MICROUSDT",
        entry_time=t0,
        exit_time=t0 + timedelta(hours=4),
        entry_price=1.0,
        stop_price=0.90,
        exit_price=3.0,
        risk_pct=10.0,
        net_pnl_pct=200.0,
        r_mult=20.0,
        worst_low_price=0.95,
        bar_quote_vol_usd=20_000.0,
        trailing_24h_qv_usd=5_000_000.0,
    )
    tr_crash = bt_utils.PortfolioTradeInput(
        symbol="CRASHUSDT",
        entry_time=t0 + timedelta(hours=5),
        exit_time=t0 + timedelta(hours=6),
        entry_price=100.0,
        stop_price=90.0,
        exit_price=65.0,
        risk_pct=10.0,
        net_pnl_pct=-35.12,
        r_mult=-3.512,
        worst_low_price=65.0,
        bar_quote_vol_usd=50_000_000.0,
        trailing_24h_qv_usd=500_000_000.0,
    )
    port_stress = bt_utils.simulate_honest_portfolio([tr_micro, tr_crash], initial_cap=1000.0, max_positions=2, slots_3x=1)
    assert port_stress["comp_3x"]["liquidity_capped_count"] == 1
    assert port_stress["comp_3x"]["liquidated_count"] == 1
    assert port_stress["comp_3x"]["final_eq"] == pytest.approx(0.0, abs=1e-6)
    assert port_stress["headline_risk"]["final_eq"] > 1000.0


def test_no_partial_cache_write_backoff_seen_timeframe_scaling_and_365d_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # 1) Partial/failed fetch MUST NOT write truncated data into cache_dir
    cdir_fail = tmp_path / "partial_cache"
    chunk_page1 = _make_raw_klines(n=1000, bar_ms=900_000, include_forming=False)
    pages = [chunk_page1, None]  # Page 1 full (1000 bars), Page 2 fails mid-pagination!
    monkeypatch.setattr(bt_sq, "polite_get_klines", lambda params, retries=6: pages.pop(0))
    df_partial = bt_sq.fetch_klines_180d("PARTIALUSDT", interval="15m", days=10, cache_dir=cdir_fail, seed_cache_dir=None)
    assert df_partial.empty
    assert list(cdir_fail.glob("*.pkl")) == []

    # 2) HTTP 418 / 429 rate-limit backoff + WARNING log
    resp_418 = MagicMock()
    resp_418.status_code = 418
    resp_418.headers = {"Retry-After": "0", "X-MBX-USED-WEIGHT-1M": "100"}
    resp_ok = MagicMock()
    resp_ok.status_code = 200
    resp_ok.headers = {"X-MBX-USED-WEIGHT-1M": "110"}
    resp_ok.raise_for_status.return_value = None
    resp_ok.json.return_value = {"ok": True}
    seq = [resp_418, resp_ok]
    monkeypatch.setattr(sq_radar.HTTP, "get", lambda *a, **kw: seq.pop(0))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with caplog.at_level("WARNING"):
        assert sq_radar.api_get("/fapi/v1/ping") == {"ok": True}
    assert any("418" in rec.message for rec in caplog.records)

    # 3) --seen-file persistence across restarts
    seen_file = tmp_path / "watch_seen.json"
    seen = {("BTCUSDT", "2026-10-01T12:00:00+08:00"), ("SOLUSDT", "2026-10-01T12:15:00+08:00")}
    sq_radar.save_seen_state(seen_file, seen)
    assert sq_radar.load_seen_state(seen_file) == seen

    # 4) 5m / 15m / 1h timeframe windows scale proportionally from bar duration
    tf5 = sq_radar.timeframe_windows("5m")
    tf15 = sq_radar.timeframe_windows("15m")
    tf60 = sq_radar.timeframe_windows("1h")
    assert tf5["bars_24h"] == 288 and tf15["bars_24h"] == 96 and tf60["bars_24h"] == 24
    assert tf5["bars_vwap_hold"] == 24 and tf15["bars_vwap_hold"] == 8 and tf60["bars_vwap_hold"] == 2
    assert tf5["sma_5d"] == 1440 and tf15["sma_5d"] == 480 and tf60["sma_5d"] == 120
    assert tf5["max_hold_bars"] == 480 and tf15["max_hold_bars"] == 160 and tf60["max_hold_bars"] == 40

    # Verify 1h M2 detection works with only 120 bars (whereas old hardcoded 260-bar gate rejected <260 1h bars!)
    idx_1h = pd.date_range("2026-09-01", periods=120, freq="1h", tz=TPE)
    c1h = np.full(120, 20.0)
    o1h = np.full(120, 20.0)
    h1h = np.full(120, 20.05)
    l1h = np.full(120, 19.95)
    v1h = np.full(120, 2_000.0)
    qv1h = np.full(120, 40_000.0)
    tbv1h = np.full(120, 1_050.0)
    o1h[100], l1h[100], h1h[100], c1h[100] = 20.0, 19.98, 21.10, 21.05
    v1h[100], qv1h[100], tbv1h[100] = 90_000.0, 1_850_000.0, 60_000.0
    df_1h = sq_radar.compute_radar_indicators(
        pd.DataFrame(
            {"Open": o1h, "High": h1h, "Low": l1h, "Close": c1h, "Volume": v1h, "QuoteVol": qv1h, "TakerBuyVol": tbv1h},
            index=idx_1h,
        ),
        interval="1h",
    )
    hits_1h = sq_radar.scan_squeeze_breakouts(df_1h, "LINKUSDT", interval="1h")
    assert len(hits_1h) == 1 and hits_1h[0].bar_idx == 100

    # 5) Offline 365d backtest e2e from <SYMBOL>_15m_365d.pkl in --cache-dir
    cache_365 = tmp_path / "cache_365d"
    cache_365.mkdir(parents=True, exist_ok=True)
    n_bars = 650
    idx_e2e = pd.date_range("2026-09-01", periods=n_bars, freq="15min", tz=TPE)
    c = np.full(n_bars, 10.0)
    o = np.full(n_bars, 10.0)
    h = np.full(n_bars, 10.03)
    l = np.full(n_bars, 9.97)
    v = np.full(n_bars, 5_000.0)
    qv = np.full(n_bars, 50_000.0)
    tbv = np.full(n_bars, 2_600.0)

    for trig in (320, 520):
        o[trig], l[trig], h[trig], c[trig] = 10.00, 9.99, 10.55, 10.52
        v[trig], qv[trig], tbv[trig] = 220_000.0, 2_300_000.0, 150_000.0
        for k in range(trig + 1, trig + 35):
            step_px = 10.52 + (k - trig) * 0.04
            o[k], c[k], h[k], l[k] = step_px - 0.02, step_px, step_px + 0.03, step_px - 0.03
            v[k], qv[k], tbv[k] = 20_000.0, 210_000.0, 11_500.0
        # Drop below BB_MID after rally so trade closes cleanly before next setup
        for k in range(trig + 35, min(n_bars, trig + 60)):
            o[k], c[k], h[k], l[k] = 10.0, 10.0, 10.03, 9.97

    # Save using QuoteVolume column name to prove compatibility with binance-dual-engine-v8 PR #2 cache format
    pd.DataFrame(
        {
            "Open": o,
            "High": h,
            "Low": l,
            "Close": c,
            "Volume": v,
            "QuoteVolume": qv,
            "TakerBuyVol": tbv,
        },
        index=idx_e2e,
    ).to_pickle(cache_365 / "SOLUSDT_15m_365d.pkl")

    out_365 = tmp_path / "out_365d"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "backtest_180d_squeeze.py"),
            "--days", "365",
            "--cache-dir", str(cache_365),
            "--out-dir", str(out_365),
            "--min-vol", "1000000",
            "--oos-ratio", "0.30",
            "--wf-folds", "4",
            "--no-fetch-funding",
            "--skip-cards",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    html_out = (out_365 / "index.html").read_text(encoding="utf-8")
    assert "以損定倉複利" in html_out
    assert "Out-of-Sample" in html_out
    assert "Walk-Forward" in html_out
    assert "0.12%" in html_out and "0.30%" in html_out and "0.50%" in html_out
    assert "999.00" not in html_out
    assert (out_365 / "equity_curve_365d.png").is_file()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_index_html_js_repaint_and_timeframe_scaling() -> None:
    html = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
    m = re.search(r"<script>([\s\S]*?)</script>", html)
    assert m is not None
    js_body = m.group(1)
    raw = _make_raw_klines(n=275, bar_ms=900_000, include_forming=True)

    harness = f"""
const vm = require("vm");
const storage = new Map();
const sandbox = {{
  window: {{ location: {{ search: "" }} }},
  document: {{
    getElementById: () => null,
    createElement: () => ({{ className: "", innerHTML: "" }})
  }},
  localStorage: {{
    getItem: k => storage.has(k) ? storage.get(k) : null,
    setItem: (k, v) => storage.set(k, String(v))
  }},
  navigator: {{}},
  URLSearchParams,
  console,
  Date,
  Math,
  Number,
  Set,
  Array,
  JSON,
  Promise,
  isFinite,
  parseFloat,
  parseInt,
  setTimeout,
  clearInterval,
  setInterval
}};
vm.createContext(sandbox);
vm.runInContext({json.dumps(js_body)}, sandbox);

const raw = {json.dumps(raw)};
const hitsDefault = sandbox.analyzeSymbol(raw, "TESTUSDT", "15m", "sniper", 10);
const hitsForming = sandbox.analyzeSymbol(raw, "TESTUSDT", "15m", "sniper", 10, false);
const tf5 = sandbox.getTimeframeWindows("5m");
const tf15 = sandbox.getTimeframeWindows("15m");
const tf60 = sandbox.getTimeframeWindows("1h");
vm.runInContext('seenKeys.add("BTCUSDT_12345"); persistSeenKeys();', sandbox);
console.log(JSON.stringify({{
  closedLen: hitsDefault.length,
  formingLen: hitsForming.length,
  tf5_24h: tf5.bars24h,
  tf15_24h: tf15.bars24h,
  tf60_24h: tf60.bars24h,
  tf5_vwap: tf5.barsVwapHold,
  tf60_vwap: tf60.barsVwapHold,
  savedSeen: JSON.parse(storage.get("squeeze_radar_watch_seen_v1") || "[]")
}}));
"""
    proc = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["closedLen"] == 0
    assert out["formingLen"] == 1
    assert out["tf5_24h"] == 288 and out["tf15_24h"] == 96 and out["tf60_24h"] == 24
    assert out["tf5_vwap"] == 24 and out["tf60_vwap"] == 2
    assert "BTCUSDT_12345" in out["savedSeen"]

