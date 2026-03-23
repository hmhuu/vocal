import streamlit as st
import numpy as np
from scipy.fftpack import fft
from scipy.interpolate import make_interp_spline
import pandas as pd
import altair as alt
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import queue
import logging
import time

# [1] 로그 차단: webrtc 관련 내부 경고를 숨깁니다.
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)

# 2. 페이지 설정
st.set_page_config(page_title="Vocal Analyzer Pro", layout="wide")
st.title("🎤 Advanced Vocal Analyzer (2026 Optimized)")

# 3. 유틸리티 함수: 음정(C4 등) -> Hz 변환
def note_to_hz(note):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    try:
        note = note.strip().upper()
        name = note[:-1].replace('DB', 'C#').replace('EB', 'D#').replace('GB', 'F#').replace('AB', 'G#').replace('BB', 'A#')
        octave = int(note[-1])
        n = notes.index(name)
        return 440.0 * (2.0 ** ((n - 9 + (octave - 4) * 12) / 12.0))
    except:
        return None

# 4. 세션 상태 초기화
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue(maxsize=15)
if "noise_floor" not in st.session_state:
    st.session_state.noise_floor = None 
if "prev_y" not in st.session_state:
    st.session_state.prev_y = None
if "last_df" not in st.session_state:
    st.session_state.last_df = None

shared_queue = st.session_state.audio_queue

# 5. 오디오 프로세서
class OptimizedAudioProcessor(AudioProcessorBase):
    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.capture_noise = False
        self.temp_noise_floor = None

    def recv(self, frame):
        try:
            data = frame.to_ndarray().flatten()
            n = len(data)
            mag = np.abs(fft(data)[:n//2]) * (2.0 / n)
            if self.capture_noise:
                self.temp_noise_floor = mag 
                self.capture_noise = False
            if self.data_queue.full():
                self.data_queue.get_nowait()
            self.data_queue.put_nowait((mag, self.temp_noise_floor))
        except: pass
        return frame

# 6. 사이드바 및 상단 컨트롤
st.sidebar.header("⚙️ Settings")
y_max_val = st.sidebar.number_input("Intensity (Y-Max)", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
target_hz = st.sidebar.slider("Max Hz", 500, 5000, 3000)
data_points = st.sidebar.slider("Raw Data Points", 20, 150, 60)
smooth_factor = st.sidebar.slider("Smooth Factor", 50, 500, 200)
smoothing_alpha = st.sidebar.slider("Smoothing Alpha", 0.05, 1.0, 0.3)

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    ctx = webrtc_streamer(
        key="vocal-2026-final",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=lambda: OptimizedAudioProcessor(shared_queue),
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )
with c2:
    st.write("")
    # use_container_width=True 대신 width='stretch' 사용
    if st.button("🧹 Zero Noise", width='stretch'):
        if ctx.audio_processor:
            ctx.audio_processor.capture_noise = True
            st.toast("Capturing noise...")
with c3:
    st.write("")
    graph_active = st.toggle("Graph Active", value=True)
with c4:
    note_input = st.text_input("Harmony Note", value="C4")

chart_placeholder = st.empty()

# 7. 차트 생성 함수 (Altair)
def get_harmony_chart(df, target_hz, y_limit, note_str):
    base = alt.Chart(df).mark_line(color='#1E90FF', strokeWidth=2).encode(
        x=alt.X('Hz:Q', scale=alt.Scale(domain=[0, target_hz]), title='Frequency (Hz)'),
        y=alt.Y('Magnitude:Q', scale=alt.Scale(domain=[0, y_limit]), title='Magnitude')
    ).properties(height=500)

    f0 = note_to_hz(note_str)
    if f0 and f0 < target_hz:
        harmonics = [f0 * i for i in range(1, 12) if f0 * i <= target_hz]
        h_df = pd.DataFrame({'h_hz': harmonics})
        rules = alt.Chart(h_df).mark_rule(
            color='green', opacity=0.8, strokeWidth=1.5
        ).encode(x='h_hz:Q')
        return (base + rules)
    return base

# 8. 메인 분석 루프
if ctx.state.playing:
    SAMPLING_RATE = 44100
    while ctx.state.playing:
        try:
            mag, noise_floor = shared_queue.get(timeout=0.03)
            
            if graph_active:
                x_raw = np.linspace(0, target_hz, data_points)
                x_smooth = np.linspace(0, target_hz, smooth_factor)
                
                if noise_floor is not None:
                    mag = np.maximum(0, mag - noise_floor)
                
                freqs = np.fft.fftfreq(len(mag)*2, 1/SAMPLING_RATE)[:len(mag)]
                indices = np.searchsorted(freqs, x_raw)
                indices = np.clip(indices, 0, len(mag) - 1)
                y_raw = mag[indices]
                
                if st.session_state.prev_y is None or len(st.session_state.prev_y) != data_points:
                    st.session_state.prev_y = y_raw
                y_temporal = (y_raw * smoothing_alpha) + (st.session_state.prev_y * (1 - smoothing_alpha))
                st.session_state.prev_y = y_temporal
                
                spline = make_interp_spline(x_raw, y_temporal, k=3)
                y_smooth = np.maximum(0, spline(x_smooth))
                y_clipped = np.clip(y_smooth, 0, y_max_val)
                
                st.session_state.last_df = pd.DataFrame({'Hz': x_smooth, 'Magnitude': y_clipped})
            
            if st.session_state.last_df is not None:
                chart = get_harmony_chart(st.session_state.last_df, target_hz, y_max_val, note_input)
                # use_container_width=True 대신 width='stretch' 사용
                chart_placeholder.altair_chart(chart, width='stretch')
            
            time.sleep(0.001)
        except queue.Empty: continue
        except Exception: continue
else:
    if st.session_state.last_df is not None:
        chart = get_harmony_chart(st.session_state.last_df, target_hz, y_max_val, note_input)
        chart_placeholder.altair_chart(chart, width='stretch')
    st.info("🎤 Start를 눌러주세요. 경고 로그가 제거된 버전입니다.")