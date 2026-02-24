# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 20:32:31 2026
Modified for yfinance Integration & Pandas Compatibility
@author: Sanghee Han
"""

import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# -----------------------------------------------------------------------------
# 0. 라이브러리 및 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Crypto Signal Monitor & Calculator", layout="wide")

# 코인 매핑 (ccxt 형식 -> yfinance 형식)
COIN_MAP = {
    'BTC/USDT': 'BTC-USD',
    'ETH/USDT': 'ETH-USD',
    'XRP/USDT': 'XRP-USD',
    'SOL/USDT': 'SOL-USD'
}

# yfinance 지원 타임프레임 규격에 맞춤
TIMEFRAMES = {
    '15m': '15m',
    '1h': '1h',
    '2h': '1h',  # yfinance는 2h/4h 직접 지원이 불안정하여 1h 데이터를 가져와 처리하거나 1h로 대체
    '4h': '1h',
    '1d': '1d'
}

# 파일 저장 경로 (KST 기준)
DATE_STR = datetime.now().strftime("%Y-%m-%d")
FILE_NAME = f"Trading_Journal_{DATE_STR}.xlsx"

# 세션 상태 초기화
if 'status_log' not in st.session_state:
    st.session_state.status_log = []
if 'trade_plan_log' not in st.session_state:
    st.session_state.trade_plan_log = []

# -----------------------------------------------------------------------------
# 1. 데이터 가져오기 (yfinance 사용 - IP 차단 방지)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_ohlcv(symbol, timeframe):
    try:
        yf_symbol = COIN_MAP.get(symbol, 'BTC-USD')
        # 타임프레임별 데이터 기간 설정
        period = "5d" if timeframe == "15m" else "60d"
        
        df = yf.download(tickers=yf_symbol, period=period, interval=timeframe, progress=False)
        
        if df.empty: return pd.DataFrame()
        
        # 인덱스 초기화 및 컬럼 정리
        df = df.reset_index()
        # yfinance MultiIndex 컬럼 대응 및 소문자 변환
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df = df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'})
        
        # KST 변환 (UTC+9)
        df['timestamp'] = pd.to_datetime(df['timestamp']) + timedelta(hours=9)
        return df
    except Exception as e:
        st.error(f"📡 {symbol} 데이터 로드 실패: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    if df.empty or len(df) < 50: return df
    
    # BB 20, 2
    try:
        bb20 = ta.bbands(df['close'], length=20, std=2.0)
        if bb20 is not None:
            bb20.columns = ['BBL_20', 'BBM_20', 'BBU_20', 'BBB_20', 'BBP_20']
            df = pd.concat([df, bb20], axis=1)
    except: pass

    # BB 4, 4
    try:
        bb4 = ta.bbands(df['close'], length=4, std=4.0)
        if bb4 is not None:
            bb4.columns = ['BBL_4', 'BBM_4', 'BBU_4', 'BBB_4', 'BBP_4']
            df = pd.concat([df, bb4], axis=1)
    except: pass

    # CCI 20 (HLC3)
    df['CCI_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    
    # RSI & MACD
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    df['CCI_10'] = ta.cci(df['high'], df['low'], df['close'], length=10) 
    
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['MACD'] = macd['MACD_12_26_9'] 

    # Signal 5용: Donchian Channel (30봉 기준 고가/저가)
    df['High_30'] = df['high'].rolling(30).max().shift(1)
    df['Low_30']  = df['low'].rolling(30).min().shift(1)

    # ATR 14 (계산기용 필수 지표)
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df

# -----------------------------------------------------------------------------
# 2. 신호 유효성(Validity) 검사 로직 (상희님 기존 로직 유지)
# -----------------------------------------------------------------------------

def check_signal_1_div_validity(df):
    if len(df) < 50: return False, None, None
    current_idx = len(df) - 1
    for offset in range(3): 
        idx = current_idx - offset
        if idx < 20: continue
        prd = 5
        pivot_idx = idx - prd
        # Bullish
        window_low = df['low'].iloc[pivot_idx-prd : pivot_idx+prd+1]
        if df['low'].iloc[pivot_idx] == window_low.min():
            curr_low = df['low'].iloc[pivot_idx]
            for back in range(1, 30):
                prev_p_idx = pivot_idx - back
                if prev_p_idx < prd: break
                prev_win = df['low'].iloc[prev_p_idx-prd : prev_p_idx+prd+1]
                if df['low'].iloc[prev_p_idx] == prev_win.min():
                    bull_cnt = 0
                    if curr_low > df['low'].iloc[prev_p_idx]: bull_cnt += 1 
                    for key in ['RSI_14', 'CCI_10', 'MACD']:
                        if key in df.columns and df[key].iloc[pivot_idx] > df[key].iloc[prev_p_idx]: bull_cnt += 1
                    if bull_cnt >= 2: return True, df['timestamp'].iloc[idx], "Bullish"
                    break
        # Bearish
        window_high = df['high'].iloc[pivot_idx-prd : pivot_idx+prd+1]
        if df['high'].iloc[pivot_idx] == window_high.max():
            curr_high = df['high'].iloc[pivot_idx]
            for back in range(1, 30):
                prev_p_idx = pivot_idx - back
                if prev_p_idx < prd: break
                prev_win = df['high'].iloc[prev_p_idx-prd : prev_p_idx+prd+1]
                if df['high'].iloc[prev_p_idx] == prev_win.max():
                    bear_cnt = 0
                    if curr_high < df['high'].iloc[prev_p_idx]: bear_cnt += 1 
                    for key in ['RSI_14', 'CCI_10', 'MACD']:
                        if key in df.columns and df[key].iloc[pivot_idx] < df[key].iloc[prev_p_idx]: bear_cnt += 1
                    if bear_cnt >= 2: return True, df['timestamp'].iloc[idx], "Bearish"
                    break
    return False, None, None

def check_signal_2_double_bb_validity(df):
    current_idx = len(df) - 1
    for offset in range(2):
        i = current_idx - offset
        row = df.iloc[i]
        if row['close'] < row['BBL_20'] or row['close'] < row['BBL_4']:
            return True, row['timestamp'], "Oversold"
        if row['close'] > row['BBU_20'] or row['close'] > row['BBU_4']:
             return True, row['timestamp'], "Overbought"
    return False, None, None

def check_signal_3_cci_bb_validity(df):
    current_idx = len(df) - 1
    for offset in range(2):
        i = current_idx - offset
        if i < 0: continue
        row = df.iloc[i]
        if (row['CCI_20'] < -100) and (row['close'] < row['BBL_20']): return True, row['timestamp'], "Bullish"
        if (row['CCI_20'] > 100) and (row['close'] > row['BBU_20']): return True, row['timestamp'], "Bearish"
    return False, None, None

def check_signal_4_band_in_validity(df):
    curr_idx = len(df) - 1
    for offset in range(3):
        i = curr_idx - offset
        if i < 7: continue
        recent_cci = df['CCI_20'].iloc[i-6:i]
        # Bullish
        if (recent_cci < -100).sum() >= 5 and df['CCI_20'].iloc[i-1] < -100 and df['CCI_20'].iloc[i] > -100:
            return True, df['timestamp'].iloc[i], "Bullish"
        # Bearish
        if (recent_cci > 100).sum() >= 5 and df['CCI_20'].iloc[i-1] > 100 and df['CCI_20'].iloc[i] < 100:
            return True, df['timestamp'].iloc[i], "Bearish"
    return False, None, None

def check_signal_5_breakout_validity(df):
    if len(df) < 50: return False, None, None
    curr_idx = len(df) - 1
    for offset in range(5):
        i = curr_idx - offset
        if i < 10: continue
        row = df.iloc[i]
        if row['close'] > row['High_30']: return True, row['timestamp'], "Bullish Breakout"
        if row['close'] < row['Low_30']: return True, row['timestamp'], "Bearish Breakout"
    return False, None, None

# -----------------------------------------------------------------------------
# 3. 전체 스캔 실행 함수
# -----------------------------------------------------------------------------
def scan_all_status():
    status_list = []
    progress_text = "Scanning Markets (yfinance)..."
    my_bar = st.progress(0, text=progress_text)
    total_steps = len(COIN_MAP) * len(TIMEFRAMES)
    step = 0
    
    for coin_name in COIN_MAP.keys():
        for tf_key, tf_val in TIMEFRAMES.items():
            df = fetch_ohlcv(coin_name, tf_val)
            df = calculate_indicators(df)
            
            if df.empty:
                step += 1
                continue
            
            # 각 시그널 체크
            if tf_key in ['1h', '2h', '4h', '1d']:
                v1, t1, ty1 = check_signal_1_div_validity(df)
                status_list.append({"Coin": coin_name, "TF": tf_key, "Signal": "Sig 1", "Status": "Green" if v1 else "Red", "Start Time": t1 if v1 else "-", "Type": ty1 if v1 else "-"})
            
            v2, t2, ty2 = check_signal_2_double_bb_validity(df)
            status_list.append({"Coin": coin_name, "TF": tf_key, "Signal": "Sig 2", "Status": "Green" if v2 else "Red", "Start Time": t2 if v2 else "-", "Type": ty2 if v2 else "-"})

            if tf_key in ['1h', '2h', '4h', '1d']:
                v3, t3, ty3 = check_signal_3_cci_bb_validity(df)
                status_list.append({"Coin": coin_name, "TF": tf_key, "Signal": "Sig 3", "Status": "Green" if v3 else "Red", "Start Time": t3 if v3 else "-", "Type": ty3 if v3 else "-"})

            if tf_key in ['1h', '2h', '4h']:
                v4, t4, ty4 = check_signal_4_band_in_validity(df)
                status_list.append({"Coin": coin_name, "TF": tf_key, "Signal": "Sig 4", "Status": "Green" if v4 else "Red", "Start Time": t4 if v4 else "-", "Type": ty4 if v4 else "-"})
            
            v5, t5, ty5 = check_signal_5_breakout_validity(df)
            status_list.append({"Coin": coin_name, "TF": tf_key, "Signal": "Sig 5", "Status": "Green" if v5 else "Red", "Start Time": t5 if v5 else "-", "Type": ty5 if v5 else "-"})

            step += 1
            my_bar.progress(step / total_steps, text=f"Scanning {coin_name} {tf_key}...")
            
    my_bar.empty()
    return pd.DataFrame(status_list)

# -----------------------------------------------------------------------------
# 4. UI 구성 (Dashboard + Calculator)
# -----------------------------------------------------------------------------
st.title(f"🚦 Crypto Signal Monitor - {DATE_STR} (KST)")

# 상단: 실시간 시세 (yfinance)
cols = st.columns(len(COIN_MAP))
for i, coin in enumerate(COIN_MAP.keys()):
    df_brief = fetch_ohlcv(coin, '15m')
    if not df_brief.empty:
        price = df_brief['close'].iloc[-1]
        cols[i].metric(coin, f"{price:.4f}")

st.divider()

if st.button("🔄 Refresh Signals"):
    st.session_state.status_log = scan_all_status()
    st.success("Scan Updated!")

def color_status(val):
    color = '#d4edda' if val == 'Green' else '#f8d7da'
    return f'background-color: {color}; color: black'

if isinstance(st.session_state.status_log, pd.DataFrame) and not st.session_state.status_log.empty:
    df_status = st.session_state.status_log
    tab_titles = ["Dashboard", "Sig 1 (Div)", "Sig 2 (BB OR)", "Sig 3 (CCI+BB)", "Sig 4 (Band-In)", "Sig 5 (Breakout)"] + [f"{c} Multi" for c in COIN_MAP.keys()]
    tabs = st.tabs(tab_titles)
    
    with tabs[0]:
        st.subheader("📋 All Signals Status")
        # applymap 대신 map 사용 (Pandas 최신 버전 호환)
        st.dataframe(df_status.style.map(color_status, subset=['Status']), use_container_width=True)
    
    sig_names = ["Sig 1", "Sig 2", "Sig 3", "Sig 4", "Sig 5"]
    for i, sig_name in enumerate(sig_names):
        with tabs[i+1]:
            st.subheader(f"📡 {sig_name} Status")
            df_sub = df_status[df_status['Signal'] == sig_name].sort_values(by='Status', ascending=True) 
            st.dataframe(df_sub.style.map(color_status, subset=['Status']), use_container_width=True)
            
    for i, coin in enumerate(COIN_MAP.keys()):
        with tabs[6+i]:
            st.subheader(f"🔥 {coin} Multi Signals")
            df_coin = df_status[(df_status['Coin'] == coin) & (df_status['Status'] == 'Green')]
            green_count = len(df_coin)
            st.metric("Active Signals Count", f"{green_count} / 3 Required")
            if green_count >= 3:
                st.success(f"🚨 ALERT: {green_count} Signals Simultaneously Active!")
            st.dataframe(df_coin, use_container_width=True)
else:
    st.info("Please click 'Refresh Signals' to start.")

st.divider()

# -----------------------------------------------------------------------------
# 5. 레버리지 계산기
# -----------------------------------------------------------------------------
st.subheader("🧮 Position & Leverage Calculator")
calc_col1, calc_col2 = st.columns(2)

with calc_col1:
    st.markdown("##### Input Data")
    s_coin = st.selectbox("Select Coin", list(COIN_MAP.keys()))
    s_tf = st.selectbox("Reference Timeframe", ['15m', '1h', '1d'])
    rr_ratio = st.number_input("Risk/Reward Ratio (Target)", value=3.0, step=0.1, format="%.1f")
    
    df_calc = fetch_ohlcv(s_coin, s_tf)
    df_calc = calculate_indicators(df_calc)
    
    if not df_calc.empty and 'ATR_14' in df_calc.columns:
        cur_p = df_calc['close'].iloc[-1]
        cur_atr = df_calc['ATR_14'].iloc[-1]
        atr_pct = (cur_atr / cur_p) * 100
        st.info(f"Price: {cur_p:.4f} | ATR: {cur_atr:.4f} ({atr_pct:.2f}%)")
        
        sl = st.number_input("Stop Loss Price (SL)", value=0.0, format="%.4f")
        pos = "Long" if cur_p > sl else "Short"
        if sl == 0: pos = "Ready"
        st.write(f"Detected Position: **{pos}**")
        entry = st.number_input("Entry Price", value=cur_p, format="%.4f")
        
        if sl > 0:
            if (pos == "Long" and entry > sl) or (pos == "Short" and entry < sl):
                risk = abs(entry - sl)
                tp = entry + (risk * rr_ratio) if pos == "Long" else entry - (risk * rr_ratio)
                sl_dist_pct = abs(entry - sl) / entry * 100
                lev_dist = 10 / sl_dist_pct if sl_dist_pct > 0 else 0
                
                st.success("Calculations Complete!")
                st.markdown(f"- **Rec. Leverage:** `{lev_dist:.2f}x` (10% Risk)\n- **Target Price:** `{tp:.4f}`")
                
                if st.button("💾 Add to Trade Plan"):
                    st.session_state.trade_plan_log.append({
                        "Time": datetime.now().strftime("%H:%M"), "Coin": s_coin, "Type": pos,
                        "Entry": entry, "SL": sl, "TP": tp, "RR": f"1:{rr_ratio}", "Lev": round(lev_dist, 2)
                    })
            else:
                st.error("Invalid Entry vs SL.")

with calc_col2:
    st.markdown("##### 📝 Today's Trade Plan")
    if st.session_state.trade_plan_log:
        st.dataframe(pd.DataFrame(st.session_state.trade_plan_log), use_container_width=True)

if st.button("Save All Data to Excel"):
    with pd.ExcelWriter(FILE_NAME, engine='openpyxl') as writer:
        if st.session_state.status_log: pd.DataFrame(st.session_state.status_log).to_excel(writer, sheet_name='Signals', index=False)
        if st.session_state.trade_plan_log: pd.DataFrame(st.session_state.trade_plan_log).to_excel(writer, sheet_name='Plans', index=False)
    st.success(f"File saved: {FILE_NAME}")