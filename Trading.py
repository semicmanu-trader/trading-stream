# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 20:32:31 2026
Updated: 2026-02-25 (Entry Price Leverage Logic Added)
@author: Sanghee Han
"""

import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# [주의] set_page_config는 코드 최상단에 단 한 번만!
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
TIMEFRAMES = {'15m': '15m', '1h': '1h', '2h': '1h', '4h': '1h', '1d': '1d'}

DATE_STR = datetime.now().strftime("%Y-%m-%d")
FILE_NAME = f"Trading_Journal_{DATE_STR}.xlsx"

if 'status_log' not in st.session_state:
    st.session_state.status_log = []
if 'trade_plan_log' not in st.session_state:
    st.session_state.trade_plan_log = []

# -----------------------------------------------------------------------------
# 1. 데이터 가져오기 (yfinance - IP 차단 대응)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_ohlcv(symbol, timeframe):
    try:
        yf_symbol = COIN_MAP.get(symbol, 'BTC-USD')
        period = "2d" if timeframe == "15m" else "60d"
        df = yf.download(tickers=yf_symbol, period=period, interval=timeframe, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp']) + timedelta(hours=9)
        return df
    except Exception:
        return pd.DataFrame()

def calculate_indicators(df):
    if df.empty or len(df) < 50: return df
    try:
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        df['CCI_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        bb20 = ta.bbands(df['close'], length=20, std=2.0)
        if bb20 is not None:
            df['BBL_20'], df['BBU_20'] = bb20.iloc[:, 0], bb20.iloc[:, 2]
        df['High_30'] = df['high'].rolling(30).max().shift(1)
        df['Low_30']  = df['low'].rolling(30).min().shift(1)
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        return df
    except: return df

# -----------------------------------------------------------------------------
# 2. 신호 로직
# -----------------------------------------------------------------------------
def check_signals(df):
    if df.empty: return []
    last = df.iloc[-1]
    res = []
    # Sig 2: BB Out
    if last['close'] < last['BBL_20']: res.append(("Sig 2", "Green", "Oversold"))
    elif last['close'] > last['BBU_20']: res.append(("Sig 2", "Green", "Overbought"))
    # Sig 5: Breakout
    if last['close'] > last['High_30']: res.append(("Sig 5", "Green", "Bullish BO"))
    elif last['close'] < last['Low_30']: res.append(("Sig 5", "Green", "Bearish BO"))
    return res

# -----------------------------------------------------------------------------
# 3. 메인 UI
# -----------------------------------------------------------------------------
st.title(f"🚦 Crypto Signal Monitor - {DATE_STR} (KST)")

# 상단 시세
cols = st.columns(len(COIN_MAP))
for i, (name, yf_code) in enumerate(COIN_MAP.items()):
    df_p = fetch_ohlcv(name, '1h')
    if not df_p.empty:
        cols[i].metric(name, f"{df_p['close'].iloc[-1]:.4f}")

st.divider()

if st.button("🔄 Refresh Signals"):
    status_list = []
    for coin in COIN_MAP.keys():
        for tf in ['1h', '1d']:
            df = calculate_indicators(fetch_ohlcv(coin, tf))
            sigs = check_signals(df)
            for s_name, s_stat, s_type in sigs:
                status_list.append({"Coin": coin, "TF": tf, "Signal": s_name, "Status": s_stat, "Type": s_type})
    st.session_state.status_log = pd.DataFrame(status_list)

if isinstance(st.session_state.status_log, pd.DataFrame) and not st.session_state.status_log.empty:
    st.dataframe(st.session_state.status_log.style.map(lambda v: 'background-color: #d4edda' if v=='Green' else '', subset=['Status']), use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# 4. 레버리지 계산기 (진입가 입력 기능 강화)
# -----------------------------------------------------------------------------
st.subheader("🧮 Position & Leverage Calculator")
c1, c2 = st.columns(2)

with c1:
    st.markdown("##### ⚙️ Trade Parameters")
    s_coin = st.selectbox("Select Coin", list(COIN_MAP.keys()))
    df_calc = calculate_indicators(fetch_ohlcv(s_coin, '1h'))
    
    if not df_calc.empty:
        cur_p = float(df_calc['close'].iloc[-1])
        st.info(f"현재 시장가: **{cur_p:.4f}**")
        
        # [핵심 수정] 사용자가 직접 진입가를 입력할 수 있도록 추가
        entry = st.number_input("🎯 진입 가격 (Entry Price)", value=cur_p, format="%.4f", help="시장가 진입이 아닌 경우 직접 입력하세요.")
        sl = st.number_input("🛑 손절 가격 (Stop Loss)", value=entry * 0.98, format="%.4f")
        rr = st.number_input("💎 목표 손익비 (R/R Ratio)", value=3.0, step=0.1)
        
        # 계산 로직: 입력된 entry 기준으로 계산
        risk_amt = abs(entry - sl)
        if entry > 0:
            sl_pct = (risk_amt / entry) * 100
            # 권장 레버리지: 손절 시 전체 자산의 10% 손실 기준
            lev = 10 / sl_pct if sl_pct > 0 else 0
            tp = entry + (risk_amt * rr) if entry > sl else entry - (risk_amt * rr)
            
            st.success(f"**분석 결과 (진입가 {entry:.4f} 기준)**")
            st.markdown(f"""
            - **목표가 (TP):** `{tp:.4f}`
            - **손절 거리:** `{sl_pct:.2f}%`
            - **권장 레버리지:** `{lev:.1f}x` (자산 10% 리스크 기준)
            """)
            
            if st.button("💾 플랜 저장"):
                st.session_state.trade_plan_log.append({
                    "Time": datetime.now().strftime("%H:%M"), "Coin": s_coin, 
                    "Entry": entry, "SL": sl, "TP": tp, "Lev": f"{lev:.1f}x"
                })
    else:
        st.warning("데이터 로딩 중...")

with c2:
    st.markdown("##### 📝 저장된 트레이딩 플랜")
    if st.session_state.trade_plan_log:
        st.dataframe(pd.DataFrame(st.session_state.trade_plan_log), use_container_width=True)
    else:
        st.write("저장된 플랜이 없습니다.")

if st.button("Save to Excel"):
    st.success(f"파일 저장 완료: {FILE_NAME}")