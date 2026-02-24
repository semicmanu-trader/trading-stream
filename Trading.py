# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 20:32:31 2026
Full Integration: yfinance + 5 Signals + Multi-Tabs + Calculator
@author: Sanghee Han
"""

import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# [주의] set_page_config는 반드시 최상단에 딱 한 번만 나와야 합니다.
st.set_page_config(page_title="Crypto Signal Monitor & Calculator", layout="wide")

# -----------------------------------------------------------------------------
# 0. 설정 및 세션 초기화
# -----------------------------------------------------------------------------
COIN_MAP = {
    'BTC/USDT': 'BTC-USD',
    'ETH/USDT': 'ETH-USD',
    'XRP/USDT': 'XRP-USD',
    'SOL/USDT': 'SOL-USD'
}

# yfinance 규격에 맞춘 타임프레임
TIMEFRAMES = {
    '15m': '15m',
    '1h': '1h',
    '2h': '1h',  # 2h/4h는 1h 데이터를 가져와 처리
    '4h': '1h',
    '1d': '1d'
}

DATE_STR = datetime.now().strftime("%Y-%m-%d")
FILE_NAME = f"Trading_Journal_{DATE_STR}.xlsx"

if 'status_log' not in st.session_state:
    st.session_state.status_log = []
if 'trade_plan_log' not in st.session_state:
    st.session_state.trade_plan_log = []

# -----------------------------------------------------------------------------
# 1. 데이터 가져오기 (yfinance - 차단 및 MultiIndex 해결)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_ohlcv(symbol, timeframe):
    try:
        yf_symbol = COIN_MAP.get(symbol, 'BTC-USD')
        period = "2d" if timeframe == "15m" else "60d"
        df = yf.download(tickers=yf_symbol, period=period, interval=timeframe, progress=False)
        
        if df.empty: return pd.DataFrame()
        
        # yfinance MultiIndex 컬럼 평탄화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'})
        
        # KST 변환 (UTC+9)
        df['timestamp'] = pd.to_datetime(df['timestamp']) + timedelta(hours=9)
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_indicators(df):
    if df.empty or len(df) < 50: return df
    try:
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        # BB 20, 2
        bb20 = ta.bbands(df['close'], length=20, std=2.0)
        if bb20 is not None:
            df['BBL_20'] = bb20.iloc[:, 0]
            df['BBU_20'] = bb20.iloc[:, 2]

        # BB 4, 4
        bb4 = ta.bbands(df['close'], length=4, std=4.0)
        if bb4 is not None:
            df['BBL_4'] = bb4.iloc[:, 0]
            df['BBU_4'] = bb4.iloc[:, 2]

        # CCI, RSI, MACD
        df['CCI_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        df['CCI_10'] = ta.cci(df['high'], df['low'], df['close'], length=10)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['MACD'] = macd.iloc[:, 0]

        # Donchian (Signal 5)
        df['High_30'] = df['high'].rolling(30).max().shift(1)
        df['Low_30']  = df['low'].rolling(30).min().shift(1)

        # ATR (Calculator)
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        return df
    except: return df

# -----------------------------------------------------------------------------
# 2. 5가지 신호 로직 (상희님 원본 로직 복구)
# -----------------------------------------------------------------------------
def check_sig1(df): # Divergence
    if len(df) < 50: return False, None, "-"
    curr = len(df) - 1
    for off in range(3):
        idx = curr - off
        prd = 5
        piv = idx - prd
        if piv < prd: continue
        if df['low'].iloc[piv] == df['low'].iloc[piv-prd:piv+prd+1].min():
            for b in range(1, 25):
                prev = piv - b
                if prev < prd: break
                if df['low'].iloc[prev] == df['low'].iloc[prev-prd:prev+prd+1].min():
                    if df['low'].iloc[piv] > df['low'].iloc[prev] and df['RSI_14'].iloc[piv] > df['RSI_14'].iloc[prev]:
                        return True, df['timestamp'].iloc[idx], "Bullish"
    return False, None, "-"

def check_sig2(df): # Double BB
    last = df.iloc[-1]
    if last['close'] < last['BBL_20'] or last['close'] < last['BBL_4']: return True, last['timestamp'], "Oversold"
    if last['close'] > last['BBU_20'] or last['close'] > last['BBU_4']: return True, last['timestamp'], "Overbought"
    return False, None, "-"

def check_sig3(df): # CCI + BB Strict
    last = df.iloc[-1]
    if last['CCI_20'] < -100 and last['close'] < last['BBL_20']: return True, last['timestamp'], "Bullish"
    if last['CCI_20'] > 100 and last['close'] > last['BBU_20']: return True, last['timestamp'], "Bearish"
    return False, None, "-"

def check_sig4(df): # Band-In
    curr = len(df) - 1
    for off in range(3):
        idx = curr - off
        if idx < 7: continue
        recent = df['CCI_20'].iloc[idx-6:idx]
        if (recent < -100).sum() >= 5 and df['CCI_20'].iloc[idx] > -100: return True, df['timestamp'].iloc[idx], "Bullish"
        if (recent > 100).sum() >= 5 and df['CCI_20'].iloc[idx] < 100: return True, df['timestamp'].iloc[idx], "Bearish"
    return False, None, "-"

def check_sig5(df): # Breakout
    last = df.iloc[-1]
    if last['close'] > last['High_30']: return True, last['timestamp'], "Bullish BO"
    if last['close'] < last['Low_30']: return True, last['timestamp'], "Bearish BO"
    return False, None, "-"

# -----------------------------------------------------------------------------
# 3. 전체 스캔 실행
# -----------------------------------------------------------------------------
def scan_all():
    status_list = []
    prog = st.progress(0, text="Scanning Cryptos...")
    total = len(COIN_MAP) * len(TIMEFRAMES)
    step = 0
    for coin in COIN_MAP.keys():
        for tf_k, tf_v in TIMEFRAMES.items():
            df = calculate_indicators(fetch_ohlcv(coin, tf_v))
            if not df.empty:
                v1, t1, ty1 = check_sig1(df)
                status_list.append({"Coin": coin, "TF": tf_k, "Signal": "Sig 1", "Status": "Green" if v1 else "Red", "Type": ty1})
                v2, t2, ty2 = check_sig2(df)
                status_list.append({"Coin": coin, "TF": tf_k, "Signal": "Sig 2", "Status": "Green" if v2 else "Red", "Type": ty2})
                v3, t3, ty3 = check_sig3(df)
                status_list.append({"Coin": coin, "TF": tf_k, "Signal": "Sig 3", "Status": "Green" if v3 else "Red", "Type": ty3})
                v4, t4, ty4 = check_sig4(df)
                status_list.append({"Coin": coin, "TF": tf_k, "Signal": "Sig 4", "Status": "Green" if v4 else "Red", "Type": ty4})
                v5, t5, ty5 = check_sig5(df)
                status_list.append({"Coin": coin, "TF": tf_k, "Signal": "Sig 5", "Status": "Green" if v5 else "Red", "Type": ty5})
            step += 1
            prog.progress(step / total)
    prog.empty()
    return pd.DataFrame(status_list)

# -----------------------------------------------------------------------------
# 4. 메인 UI
# -----------------------------------------------------------------------------
st.title(f"🚦 Crypto Signal Monitor - {DATE_STR} (KST)")

cols = st.columns(len(COIN_MAP))
for i, coin in enumerate(COIN_MAP.keys()):
    df_p = fetch_ohlcv(coin, '1h')
    if not df_p.empty: cols[i].metric(coin, f"{df_p['close'].iloc[-1]:.4f}")

st.divider()

if st.button("🔄 Refresh Signals"):
    st.session_state.status_log = scan_all()
    st.success("Scan Updated!")

def color_status(val):
    color = '#d4edda' if val == 'Green' else '#f8d7da'
    return f'background-color: {color}; color: black'

if isinstance(st.session_state.status_log, pd.DataFrame) and not st.session_state.status_log.empty:
    df_status = st.session_state.status_log
    tabs = st.tabs(["Dashboard", "Sig 1", "Sig 2", "Sig 3", "Sig 4", "Sig 5"] + [f"{c} Multi" for c in COIN_MAP.keys()])
    
    with tabs[0]:
        st.subheader("📋 All Signals")
        st.dataframe(df_status.style.map(color_status, subset=['Status']), use_container_width=True)
    
    for i in range(1, 6):
        with tabs[i]:
            sig_name = f"Sig {i}"
            df_sub = df_status[df_status['Signal'] == sig_name].sort_values(by='Status')
            st.dataframe(df_sub.style.map(color_status, subset=['Status']), use_container_width=True)
            
    for i, coin in enumerate(COIN_MAP.keys()):
        with tabs[6+i]:
            df_coin = df_status[(df_status['Coin'] == coin) & (df_status['Status'] == 'Green')]
            st.metric("Active Signals", f"{len(df_coin)} / 3 Req")
            if len(df_coin) >= 3: st.success("🚨 ALERT: High Confluence!")
            st.dataframe(df_coin, use_container_width=True)
else:
    st.info("Click 'Refresh Signals' to start.")

st.divider()

# -----------------------------------------------------------------------------
# 5. 레버리지 계산기
# -----------------------------------------------------------------------------
st.subheader("🧮 Position & Leverage Calculator")
c1, c2 = st.columns(2)
with c1:
    s_coin = st.selectbox("Select Coin", list(COIN_MAP.keys()))
    df_calc = calculate_indicators(fetch_ohlcv(s_coin, '1h'))
    if not df_calc.empty:
        cur_p = float(df_calc['close'].iloc[-1])
        st.info(f"Price: {cur_p:.4f}")
        sl = st.number_input("Stop Loss (SL)", value=cur_p * 0.98, format="%.4f")
        rr = st.number_input("RR Ratio", value=3.0, step=0.1)
        risk = abs(cur_p - sl)
        tp = cur_p + (risk * rr) if cur_p > sl else cur_p - (risk * rr)
        sl_pct = (risk / cur_p) * 100
        lev = 10 / sl_pct if sl_pct > 0 else 0
        st.success(f"TP: {tp:.4f} | Rec. Leverage: {lev:.1f}x")
        if st.button("💾 Save Plan"):
            st.session_state.trade_plan_log.append({
                "Time": datetime.now().strftime("%H:%M"), "Coin": s_coin, "Entry": cur_p, "SL": sl, "TP": tp, "Lev": round(lev, 1)
            })

with c2:
    st.markdown("##### 📝 Trade Plan Log")
    if st.session_state.trade_plan_log:
        st.dataframe(pd.DataFrame(st.session_state.trade_plan_log), use_container_width=True)