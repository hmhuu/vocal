import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.fftpack import fft
from scipy.interpolate import make_interp_spline
import pandas as pd
import altair as alt
import time

# 1. 페이지 설정
st.set_page_config(page_title="Vocal Analyzer Lite", layout="wide")
st.markdown("""<style> .main { overflow: hidden; } div.block-container { padding-top: 1.5rem; } </style>""", unsafe_allow_html=True)
st.title("🚀 Ultra-Light Vocal Analyzer")

# 2. 사이드바 (프레임 확보를 위한 가이드라인 설정)
st.sidebar.header("⚡ Performance Tuning")
y_limit = st.sidebar.number_input("Y-Max", min_value=0.0001, value=1.0, format="%.4f")
target_hz = st.sidebar.number_input("X-Max (Hz)", min_value=500, value=3000)

# [프레임 최적화 핵심] 데이터 포인트를 줄일수록 60fps에 가까워집니다.
data_points = st.sidebar.slider("Raw Points", 10, 100, 40) 
smooth_factor = st.sidebar.slider("Smooth Factor", 20, 200, 60)
smoothing_alpha = st.sidebar.slider("Responsiveness", 0.1, 1.0, 0.6)

if 'analyzing' not in st.session_state: st.session_state.analyzing = True
if 'last_df' not in st.session_state:
    st.session_state.last_df = pd.DataFrame({'Hz': np.linspace(0, target_hz, smooth_factor), 'Magnitude': 0})
if 'noise_floor' not in st.session_state: st.session_state.noise_floor = np.zeros(data_points)
if 'prev_y' not in st.session_state: st.session_state.prev_y = np.zeros(data_points)

# 오디오 설정
SAMPLING_RATE = 44100
CHUNK = 1024 

def note_to_hz(note):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    try:
        name = note[:-1].upper().replace('DB', 'C#').replace('EB', 'D#').replace('GB', 'F#').replace('AB', 'G#').replace('BB', 'A#')
        octave = int(note[-1])
        n = notes.index(name)
        return 440.0 * (2.0 ** ((n - 9 + (octave - 4) * 12) / 12.0))
    except: return None

# 3. UI 컨트롤
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    if st.button("⏹️ Stop" if st.session_state.analyzing else "🔴 Run", width='stretch'):
        st.session_state.analyzing = not st.session_state.analyzing
        st.rerun()
with c2:
    if st.button("🧹 Zero Noise", width='stretch'):
        st.session_state.capture_noise = True
with c3:
    note_input = st.text_input("Harmony Note", value="C4")

chart_placeholder = st.empty()

# 4. [핵심 최적화] 초경량 차트 생성 함수
def get_ultralight_chart(df, target_hz, y_limit, note_input):
    # Axis 객체를 통해 모든 장식 요소 제거 (grid, labels, ticks = False)
    no_axis = alt.Axis(grid=False, labels=False, ticks=False, title=None)

    base = alt.Chart(df).mark_line(
        color='#1E90FF', strokeWidth=2, clip=True, interpolate='linear'
    ).encode(
        x=alt.X('Hz:Q', scale=alt.Scale(domain=[0, target_hz], nice=False), axis=no_axis),
        y=alt.Y('Magnitude:Q', scale=alt.Scale(domain=[0, y_limit], nice=False), axis=no_axis)
    ).properties(height=500)

    harmony_hz = note_to_hz(note_input)
    if harmony_hz and harmony_hz < target_hz:
        h_vals = [harmony_hz * i for i in range(1, 11) if harmony_hz * i <= target_hz]
        rules = alt.Chart(pd.DataFrame({'h': h_vals})).mark_rule(
            color='green', opacity=0.8, strokeWidth=1.5
        ).encode(x='h:Q')
        return (base + rules)
    return base

# 5. 메인 루프
if st.session_state.analyzing:
    x_raw = np.linspace(0, target_hz, data_points)
    x_smooth = np.linspace(0, target_hz, smooth_factor)
    
    while st.session_state.analyzing:
        try:
            recording = sd.rec(CHUNK, samplerate=SAMPLING_RATE, channels=1, blocking=True)
            audio_data = recording.flatten()
            
            fft_mag = np.abs(fft(audio_data)[:CHUNK//2]) * (2.0 / CHUNK)
            freqs = np.fft.fftfreq(CHUNK, 1/SAMPLING_RATE)[:CHUNK//2]
            
            indices = np.digitize(freqs, np.linspace(0, target_hz, data_points + 1))
            y_curr = np.zeros(data_points)
            for i in range(data_points):
                mask = (indices == i + 1)
                if np.any(mask): y_curr[i] = np.mean(fft_mag[mask])

            if st.session_state.get('capture_noise', False):
                st.session_state.noise_floor = y_curr.copy()
                st.session_state.capture_noise = False
            
            y_denoised = np.maximum(0, y_curr - st.session_state.noise_floor)
            y_visual = (st.session_state.prev_y * (1 - smoothing_alpha)) + (y_denoised * smoothing_alpha)
            st.session_state.prev_y = y_visual

            spline = make_interp_spline(x_raw, y_visual, k=3)
            y_final = np.maximum(0, spline(x_smooth))
            
            st.session_state.last_df = pd.DataFrame({'Hz': x_smooth, 'Magnitude': y_final})
            chart = get_ultralight_chart(st.session_state.last_df, target_hz, y_limit, note_input)
            
            chart_placeholder.altair_chart(chart, width='stretch')
            # 60fps 근접을 위해 sleep을 완전히 제거하거나 0에 가깝게 설정
            time.sleep(0.0001)

        except Exception:
            break
else:
    chart = get_ultralight_chart(st.session_state.last_df, target_hz, y_limit, note_input)
    chart_placeholder.altair_chart(chart, width='stretch')