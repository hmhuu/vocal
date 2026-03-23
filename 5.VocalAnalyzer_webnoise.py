import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.fftpack import fft
from scipy.interpolate import make_interp_spline
import pandas as pd
import altair as alt
import time

# 1. 페이지 설정
st.set_page_config(page_title="Vocal Analyzer Web", layout="wide")
st.title("🎤 Vocal Analyzer (Seamless Toggle)")

# 2. 사이드바 설정
st.sidebar.header("⚙️ Professional Settings")
y_limit = st.sidebar.number_input("Intensity (Y-Max)", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
target_hz = st.sidebar.number_input("Hz MAX (X-Max)", min_value=500, value=3000, step=500)
data_points = st.sidebar.slider("Data Points (Raw)", 20, 300, 80)
smooth_factor = st.sidebar.slider("Smooth Factor (Interpolation)", 50, 500, 200)
smoothing_alpha = st.sidebar.slider("Smoothing Alpha (Time Decay)", 0.01, 1.0, 0.3)

# [초기 설정]
if 'analyzing' not in st.session_state:
    st.session_state.analyzing = True
if 'last_df' not in st.session_state:
    # 초기 빈 데이터 생성 (Hz 0~target_hz까지 0값으로 채움)
    st.session_state.last_df = pd.DataFrame({'Hz': np.linspace(0, target_hz, smooth_factor), 'Magnitude': 0})
if 'noise_floor' not in st.session_state:
    st.session_state.noise_floor = np.zeros(data_points)

# 설정값 변경 시 배열 초기화
if "last_data_points" not in st.session_state or st.session_state.last_data_points != data_points:
    st.session_state.prev_y = np.zeros(data_points)
    st.session_state.noise_floor = np.zeros(data_points)
    st.session_state.last_data_points = data_points

SAMPLING_RATE = 44100
CHUNK = 2048 

def note_to_hz(note):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    try:
        name = note[:-1].upper().replace('DB', 'C#').replace('EB', 'D#').replace('GB', 'F#').replace('AB', 'G#').replace('BB', 'A#')
        octave = int(note[-1])
        n = notes.index(name)
        return 440.0 * (2.0 ** ((n - 9 + (octave - 4) * 12) / 12.0))
    except: return None

# 3. 분석 및 시각화 공통 함수
def get_final_chart(df, target_hz, y_limit, note_input):
    harmony_hz = note_to_hz(note_input)
    harmonics = []
    if harmony_hz and harmony_hz < target_hz:
        for i in range(1, 11):
            if harmony_hz * i > target_hz: break
            harmonics.append(harmony_hz * i)

    base = alt.Chart(df).mark_line(color='#1E90FF', strokeWidth=2, clip=True).encode(
        x=alt.X('Hz:Q', scale=alt.Scale(domain=[0, target_hz], clamp=True), title='Frequency (Hz)'),
        y=alt.Y('Magnitude:Q', scale=alt.Scale(domain=[0, y_limit], nice=False, clamp=True), title='Magnitude')
    )

    if harmonics:
        h_df = pd.DataFrame({'h_hz': harmonics})
        rules = alt.Chart(h_df).mark_rule(color='green', opacity=0.8, strokeWidth=1.5).encode(x='h_hz:Q')
        return (base + rules)
    return base

# UI 컨트롤
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    button_label = "⏹️ Stop (Pause)" if st.session_state.analyzing else "🔴 Run / Restart"
    if st.button(button_label, use_container_width=True):
        st.session_state.analyzing = not st.session_state.analyzing
        st.rerun()

with c2:
    if st.button("🧹 Set Noise Floor", use_container_width=True):
        st.session_state.capture_noise = True
        st.toast("노이즈 캡처 준비 중...")

with c3:
    note_input = st.text_input("Harmony Note (e.g. C4)", value="C4")

# [핵심] 차트 출력 영역 선언 후 즉시 마지막 상태 렌더링 (끊김 방지)
chart_placeholder = st.empty()
initial_chart = get_final_chart(st.session_state.last_df, target_hz, y_limit, note_input)
chart_placeholder.altair_chart(initial_chart.properties(height=550), width='stretch')

# 4. 메인 로직
if st.session_state.analyzing:
    x_raw = np.linspace(0, target_hz, data_points)
    x_smooth = np.linspace(0, target_hz, smooth_factor)
    
    while st.session_state.analyzing:
        try:
            recording = sd.rec(CHUNK, samplerate=SAMPLING_RATE, channels=1, blocking=True)
            audio_data = recording.flatten()
            
            fft_mag = np.abs(fft(audio_data)[:CHUNK//2])
            freqs = np.fft.fftfreq(CHUNK, 1/SAMPLING_RATE)[:CHUNK//2]
            
            indices = np.digitize(freqs, np.linspace(0, target_hz, data_points + 1))
            y_curr = np.zeros(data_points)
            for i in range(data_points):
                mask = (indices == i + 1)
                if np.any(mask):
                    val = np.mean(fft_mag[mask]) * (2.0 / CHUNK)
                    y_curr[i] = max(0, val)

            # 노이즈 캡처 및 차감
            if st.session_state.get('capture_noise', False):
                st.session_state.noise_floor = y_curr.copy()
                st.session_state.capture_noise = False
            
            y_denoised = np.maximum(0, y_curr - st.session_state.noise_floor)

            if len(st.session_state.prev_y) != data_points:
                st.session_state.prev_y = np.zeros(data_points)

            y_visual = (st.session_state.prev_y * (1 - smoothing_alpha)) + (y_denoised * smoothing_alpha)
            st.session_state.prev_y = y_visual

            spline = make_interp_spline(x_raw, y_visual, k=3)
            y_smooth = np.maximum(0, spline(x_smooth))
            
            # 데이터프레임 업데이트 및 저장
            st.session_state.last_df = pd.DataFrame({'Hz': x_smooth, 'Magnitude': y_smooth})
            
            # 실시간 차트 업데이트
            current_chart = get_final_chart(st.session_state.last_df, target_hz, y_limit, note_input)
            chart_placeholder.altair_chart(current_chart.properties(height=550), width='stretch')
            
            time.sleep(0.001)

        except Exception as e:
            st.session_state.analyzing = False
            break
else:
    st.info("⏸️ 분석이 일시정지되었습니다. 마지막 파형을 고정 중입니다.")