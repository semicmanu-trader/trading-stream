# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 20:32:31 2026

@author: Sanghee Han
"""

import streamlit as st
import time
from datetime import datetime, timedelta
import os
import numpy as np

# -----------------------------------------------------------------------------
# 0. 라이브러리 및 설정
# -----------------------------------------------------------------------------
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta
    import openpyxl
except ImportError as e:
    st.error(f"❌ 필수 라이브러리가 설치되지 않았습니다. 터미널에 입력하세요: pip install ccxt pandas pandas_ta openpyxl")
    st.stop()

st.set_page_config(page_title="Crypto Signal Monitor & Calculator", layout="wide")

# 코인 및 타임프레임
COINS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT']
TIMEFRAMES = {
    '15m': '15m',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h',
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

exchange = ccxt.binance()

# -----------------------------------------------------------------------------
# 1. 데이터 가져오기 (KST 변환)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_ohlcv(symbol, timeframe, limit=1000):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # KST 변환 (UTC+9)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + timedelta(hours=9)
        return df
    except Exception:
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
    
    # Signal 1용 지표 (Divergence)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    df['CCI_10'] = ta.cci(df['high'], df['low'], df['close'], length=10) 
    
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['MACD'] = macd['MACD_12_26_9'] 

    # Signal 5용: Donchian Channel (30봉 기준 고가/저가)
    # shift(1)을 하여 '현재 봉을 제외한' 과거 30봉의 고가/저가를 구함
    df['High_30'] = df['high'].rolling(30).max().shift(1)
    df['Low_30']  = df['low'].rolling(30).min().shift(1)

    # ATR 14 (계산기용 필수 지표)
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df

# -----------------------------------------------------------------------------
# 2. 신호 유효성(Validity) 검사 로직
# -----------------------------------------------------------------------------

def check_signal_1_div_validity(df):
    """ Signal 1: Divergence (3중 2 컨플루언스, 3봉 유효) """
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
                    if curr_low > df['low'].iloc[prev_p_idx]: bull_cnt += 1 # Hidden
                    for key in ['RSI_14', 'CCI_10', 'MACD']:
                        if df[key].iloc[pivot_idx] > df[key].iloc[prev_p_idx]: bull_cnt += 1
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
                    if curr_high < df['high'].iloc[prev_p_idx]: bear_cnt += 1 # Hidden
                    for key in ['RSI_14', 'CCI_10', 'MACD']:
                        if df[key].iloc[pivot_idx] < df[key].iloc[prev_p_idx]: bear_cnt += 1
                    if bear_cnt >= 2: return True, df['timestamp'].iloc[idx], "Bearish"
                    break

    return False, None, None

def check_signal_2_double_bb_validity(df):
    """ Signal 2: Double BB (OR Condition, 2봉 유효) """
    current_idx = len(df) - 1
    for offset in range(2): # 0, 1
        i = current_idx - offset
        row = df.iloc[i]
        
        # Oversold: Close < BBL_20 OR Close < BBL_4
        if row['close'] < row['BBL_20'] or row['close'] < row['BBL_4']:
            return True, row['timestamp'], "Oversold"
            
        # Overbought: Close > BBU_20 OR Close > BBU_4
        if row['close'] > row['BBU_20'] or row['close'] > row['BBU_4']:
             return True, row['timestamp'], "Overbought"
             
    return False, None, None

def check_signal_3_cci_bb_validity(df):
    """ Signal 3: CCI + BB (Strict, 1봉 유효) """
    current_idx = len(df) - 1
    for offset in range(2): # 0, 1
        i = current_idx - offset
        if i < 0: continue
        row = df.iloc[i]
        
        is_bull = (row['CCI_20'] < -100) and (row['close'] < row['BBL_20'])
        is_bear = (row['CCI_20'] > 100) and (row['close'] > row['BBU_20'])
        
        if is_bull: return True, row['timestamp'], "Bullish"
        if is_bear: return True, row['timestamp'], "Bearish"
        
    return False, None, None

def check_signal_4_band_in_validity(df):
    """ Signal 4: Band-In (3봉 유효, 이탈 시 리셋) """
    curr_idx = len(df) - 1
    for offset in range(3): # 0, 1, 2
        i = curr_idx - offset
        if i < 7: continue
        
        recent_cci = df['CCI_20'].iloc[i-6:i]
        
        # Bullish
        was_os = (recent_cci < -100).sum() >= 5
        now_in = df['CCI_20'].iloc[i] > -100
        prev_was_os = df['CCI_20'].iloc[i-1] < -100
        if was_os and prev_was_os and now_in:
            is_reset = False
            for k in range(i + 1, curr_idx + 1):
                if df['CCI_20'].iloc[k] < -100: is_reset = True
            if not is_reset: return True, df['timestamp'].iloc[i], "Bullish"

        # Bearish
        was_ob = (recent_cci > 100).sum() >= 5
        now_in_bear = df['CCI_20'].iloc[i] < 100
        prev_was_ob = df['CCI_20'].iloc[i-1] > 100
        if was_ob and prev_was_ob and now_in_bear:
            is_reset = False
            for k in range(i + 1, curr_idx + 1):
                if df['CCI_20'].iloc[k] > 100: is_reset = True
            if not is_reset: return True, df['timestamp'].iloc[i], "Bearish"

    return False, None, None

def check_signal_5_breakout_validity(df):
    """
    Signal 5: 30-Candle Breakout
    - Green 조건: 최근 5봉 이내에 돌파(Breakout)가 발생.
    - 유지 조건: 고점을 갱신하면(재돌파) 5봉 타이머 리셋(연장).
    - Red(Invalid) 조건: 돌파 후 가격이 기준선 아래로 내려와서 **2봉 연속** 회복 못하면 무효.
    """
    if len(df) < 50: return False, None, None
    curr_idx = len(df) - 1
    
    # 최근 5개 봉(0~4)을 검사 (가장 최근 돌파를 찾음)
    for offset in range(5):
        i = curr_idx - offset # 검사할 시점 (과거 -> 현재)
        if i < 10: continue
        
        row = df.iloc[i]
        
        # 1. Bullish Breakout Check
        breakout_level_high = row['High_30']
        if row['close'] > breakout_level_high:
            # i 시점에 돌파 발생!
            # i 이후부터 현재(curr_idx)까지 '2봉 연속 이탈'이 있었는지 검사
            
            fail_count = 0
            is_invalidated = False
            
            # i+1 부터 현재까지 순회
            for k in range(i + 1, curr_idx + 1):
                # 돌파 레벨 밑으로 내려갔는가?
                # (주의: 고점 갱신 시 레벨이 올라갈 수 있으나, 여기선 최초 돌파 시점의 레벨 or 
                #  그 시점 기준의 레벨로 단순화. 더 정확히는 매 봉마다 기준선이 변하지만 
                #  '재돌파'하면 앞선 루프(offset이 더 작은)에서 잡히므로 여기선 단순 유지여부만 봄)
                
                # 재돌파(갱신) 했다면? -> fail_count 초기화 (유지됨)
                if df['close'].iloc[k] > breakout_level_high:
                    fail_count = 0
                # 이탈 했다면? -> 카운트 증가
                elif df['close'].iloc[k] < breakout_level_high:
                    fail_count += 1
                
                # 2봉 연속 이탈 시 무효화
                if fail_count >= 2:
                    is_invalidated = True
                    break
            
            if not is_invalidated:
                return True, row['timestamp'], "Bullish Breakout"

        # 2. Bearish Breakout Check
        breakout_level_low = row['Low_30']
        if row['close'] < breakout_level_low:
            fail_count = 0
            is_invalidated = False
            
            for k in range(i + 1, curr_idx + 1):
                if df['close'].iloc[k] < breakout_level_low: # 갱신(더 떨어짐)
                    fail_count = 0
                elif df['close'].iloc[k] > breakout_level_low: # 이탈(반등)
                    fail_count += 1
                
                if fail_count >= 2:
                    is_invalidated = True
                    break
            
            if not is_invalidated:
                return True, row['timestamp'], "Bearish Breakout"

    return False, None, None

# -----------------------------------------------------------------------------
# 3. 전체 스캔 실행 함수
# -----------------------------------------------------------------------------
def scan_all_status():
    status_list = []
    
    progress_text = "Scanning Markets..."
    my_bar = st.progress(0, text=progress_text)
    total_steps = len(COINS) * len(TIMEFRAMES)
    step = 0
    
    for coin in COINS:
        for tf_key, tf_val in TIMEFRAMES.items():
            df = fetch_ohlcv(coin, tf_val, limit=200)
            df = calculate_indicators(df)
            
            if df.empty:
                step += 1
                continue
            
            # Sig 1
            if tf_key in ['1h', '2h', '4h', '1d']:
                valid, time, type_ = check_signal_1_div_validity(df)
                status_list.append({"Coin": coin, "TF": tf_key, "Signal": "Sig 1", "Status": "Green" if valid else "Red", "Start Time": time if valid else "-", "Type": type_ if valid else "-"})
            
            # Sig 2 (OR)
            if tf_key in ['15m', '1h', '2h', '4h', '1d']:
                valid, time, type_ = check_signal_2_double_bb_validity(df)
                status_list.append({"Coin": coin, "TF": tf_key, "Signal": "Sig 2", "Status": "Green" if valid else "Red", "Start Time": time if valid else "-", "Type": type_ if valid else "-"})

            # Sig 3
            if tf_key in ['2h', '4h', '1d']:
                valid, time, type_ = check_signal_3_cci_bb_validity(df)
                status_list.append({"Coin": coin, "TF": tf_key, "Signal": "Sig 3", "Status": "Green" if valid else "Red", "Start Time": time if valid else "-", "Type": type_ if valid else "-"})

            # Sig 4
            if tf_key in ['1h', '2h', '4h']:
                valid, time, type_ = check_signal_4_band_in_validity(df)
                status_list.append({"Coin": coin, "TF": tf_key, "Signal": "Sig 4", "Status": "Green" if valid else "Red", "Start Time": time if valid else "-", "Type": type_ if valid else "-"})
            
            # Sig 5 (Breakout, 5 bars valid, 2 bars fail)
            if tf_key in ['15m', '1h', '2h', '4h', '1d']:
                valid, time, type_ = check_signal_5_breakout_validity(df)
                status_list.append({"Coin": coin, "TF": tf_key, "Signal": "Sig 5", "Status": "Green" if valid else "Red", "Start Time": time if valid else "-", "Type": type_ if valid else "-"})

            step += 1
            my_bar.progress(step / total_steps, text=f"Scanning {coin} {tf_key}...")
            
    my_bar.empty()
    return pd.DataFrame(status_list)

# -----------------------------------------------------------------------------
# 4. UI 구성 (Dashboard + Calculator)
# -----------------------------------------------------------------------------
st.title(f"🚦 Crypto Signal Monitor - {DATE_STR} (KST)")

# 상단: 실시간 시세
cols = st.columns(len(COINS))
for i, coin in enumerate(COINS):
    df = fetch_ohlcv(coin, '15m', limit=5)
    if not df.empty:
        price = df['close'].iloc[-1]
        cols[i].metric(coin, f"{price:.4f}")

st.divider()

# 스캔 버튼
if st.button("🔄 Refresh Signals"):
    st.session_state.status_log = scan_all_status()
    st.success("Scan Updated!")

# 스타일링 (Green/Red)
def color_status(val):
    color = '#d4edda' if val == 'Green' else '#f8d7da' # 연한 초록 / 연한 빨강
    return f'background-color: {color}; color: black'

if isinstance(st.session_state.status_log, pd.DataFrame) and not st.session_state.status_log.empty:
    df_status = st.session_state.status_log
    
    # 탭 구성: Dashboard + Signals(1~5) + Multi(Coins)
    tab_titles = ["Dashboard", "Sig 1 (Div)", "Sig 2 (BB OR)", "Sig 3 (CCI+BB)", "Sig 4 (Band-In)", "Sig 5 (Breakout)"] + [f"{c} Multi" for c in COINS]
    tabs = st.tabs(tab_titles)
    
    # [Tab 0] Dashboard
    with tabs[0]:
        st.subheader("📋 All Signals Status")
        st.dataframe(df_status.style.applymap(color_status, subset=['Status']), use_container_width=True)
    
    # [Tab 1~5] Individual Signals
    sig_names = ["Sig 1", "Sig 2", "Sig 3", "Sig 4", "Sig 5"]
    for i, sig_name in enumerate(sig_names):
        with tabs[i+1]:
            st.subheader(f"📡 {sig_name} Status")
            
            # 설명 추가
            if sig_name == "Sig 2":
                st.caption("ℹ️ Condition: Close > Any BB Upper OR Close < Any BB Lower (OR Logic)")
            if sig_name == "Sig 5":
                st.caption("ℹ️ Breakout of 10-Candle High/Low (Valid 5 bars, Reset on new high, Invalid if 2 bars deviate)")

            df_sub = df_status[df_status['Signal'] == sig_name].sort_values(by='Status', ascending=True) 
            st.dataframe(df_sub.style.applymap(color_status, subset=['Status']), use_container_width=True)
            
    # [Tab 6~] Multi Signals per Coin
    for i, coin in enumerate(COINS):
        with tabs[6+i]:
            st.subheader(f"🔥 {coin} Multi Signals (3+ Active)")
            df_coin = df_status[(df_status['Coin'] == coin) & (df_status['Status'] == 'Green')]
            green_count = len(df_coin)
            
            st.metric("Active Signals Count", f"{green_count} / 3 Required")
            
            if green_count >= 3:
                st.success(f"🚨 ALERT: {green_count} Signals Simultaneously Active!")
                st.dataframe(df_coin, use_container_width=True)
            else:
                st.info("Not enough active signals (Need 3+).")
                if green_count > 0:
                    st.write("Currently Active:")
                    st.dataframe(df_coin, use_container_width=True)
else:
    st.info("Please click 'Refresh Signals' to start.")

st.divider()

# -----------------------------------------------------------------------------
# 5. 레버리지 계산기 (손익비 설정 추가)
# -----------------------------------------------------------------------------
st.subheader("🧮 Position & Leverage Calculator")

calc_col1, calc_col2 = st.columns(2)

with calc_col1:
    st.markdown("##### Input Data")
    s_coin = st.selectbox("Select Coin", COINS)
    s_tf = st.selectbox("Reference Timeframe", ['15m', '1h', '2h', '4h', '1d'])
    
    # 손익비(R/R Ratio) 사용자 설정
    rr_ratio = st.number_input("Risk/Reward Ratio (Target)", value=3.0, step=0.1, format="%.1f")
    
    # 데이터 가져오기
    df_calc = fetch_ohlcv(s_coin, s_tf, limit=50)
    df_calc = calculate_indicators(df_calc)
    
    if not df_calc.empty and 'ATR_14' in df_calc.columns:
        cur_p = df_calc['close'].iloc[-1]
        cur_atr = df_calc['ATR_14'].iloc[-1]
        
        if pd.isna(cur_atr) or cur_atr == 0: cur_atr = cur_p * 0.01 
        atr_pct = (cur_atr / cur_p) * 100
        
        st.info(f"Price: {cur_p:.4f} | ATR: {cur_atr:.4f} ({atr_pct:.2f}%)")
        
        sl = st.number_input("Stop Loss Price (SL)", value=0.0, format="%.4f")
        pos = "Long" if cur_p > sl else "Short"
        if sl == 0: pos = "Ready"
        
        st.write(f"Detected Position: **{pos}**")
        entry = st.number_input("Entry Price", value=cur_p, format="%.4f")
        
        valid_input = False
        if sl > 0:
            if (pos == "Long" and entry > sl) or (pos == "Short" and entry < sl):
                valid_input = True
            else:
                st.error("Invalid Entry: Long entry must be > SL, Short entry must be < SL.")
        
        if valid_input:
            # 레버리지 계산
            lev_vol = 10 / atr_pct if atr_pct > 0 else 0
            sl_dist_pct = abs(entry - sl) / entry * 100
            lev_dist = 10 / sl_dist_pct if sl_dist_pct > 0 else 0
            
            # 목표가 계산 (사용자 설정 R/R 적용)
            risk = abs(entry - sl)
            tp = entry + (risk * rr_ratio) if pos == "Long" else entry - (risk * rr_ratio)
            
            st.success("Calculations Complete!")
            st.markdown(f"""
            - **Rec. Leverage (Volatility):** `{lev_vol:.2f}x`
            - **Rec. Leverage (SL Dist):** `{lev_dist:.2f}x`
            - **Target Price (1:{rr_ratio}):** `{tp:.4f}`
            """)
            
            feedback = st.text_area("Trading Notes / Feedback")
            
            if st.button("💾 Add to Trade Plan"):
                plan_entry = {
                    "Time": datetime.now().strftime("%H:%M"),
                    "Coin": s_coin,
                    "Type": pos,
                    "Entry": entry,
                    "SL": sl,
                    "TP": tp,
                    "RR": f"1:{rr_ratio}",
                    "Lev(Vol)": round(lev_vol, 2),
                    "Lev(Dist)": round(lev_dist, 2),
                    "Feedback": feedback
                }
                st.session_state.trade_plan_log.append(plan_entry)
                st.success("Plan saved to list!")

with calc_col2:
    st.markdown("##### 📝 Today's Trade Plan")
    if st.session_state.trade_plan_log:
        st.dataframe(pd.DataFrame(st.session_state.trade_plan_log), use_container_width=True)
    else:
        st.write("No plans added yet.")

# 파일 저장 버튼
if st.button("Save All Data to Excel"):
    with pd.ExcelWriter(FILE_NAME, engine='openpyxl') as writer:
        if isinstance(st.session_state.status_log, pd.DataFrame):
            st.session_state.status_log.to_excel(writer, sheet_name='Signal_Status', index=False)
        if st.session_state.trade_plan_log:
            pd.DataFrame(st.session_state.trade_plan_log).to_excel(writer, sheet_name='Trade_Plan', index=False)
    st.success(f"File saved: {FILE_NAME}")