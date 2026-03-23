import streamlit as st
import numpy as np
from scipy.fftpack import fft
import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import queue
import logging

# [핵심] 터미널 경고 로그 차단
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)

# 1. 페이지 설정
st.set_page_config(page_title="Vocal Analyzer Pro", layout="wide")
st.title("🎤 Advanced Vocal Analyzer")

# 2. 세션 상태 및 큐 초기화
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue(maxsize=15)
if "noise_floor" not in st.session_state:
    st.session_state.noise_floor = None 

shared_queue = st.session_state.audio_queue

# 3. 오디오 프로세서
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
        except:
            pass
        return frame

# 4. 사이드바 설정 (data_points 추가)
st.sidebar.header("⚙️ Settings")
y_max_val = st.sidebar.number_input("Intensity (Y-Max)", min_value=0.0001, value=10.0, step=0.1, format="%.4f")
target_hz = st.sidebar.slider("Max Hz", 500, 5000, 3000)

# 요청하신 data_points 변수 추가 (기본값 80)
data_points = st.sidebar.slider("Data Points (Resolution)", min_value=10, max_value=300, value=80, step=10)

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🧹 Zero Noise", use_container_width=True):
        st.session_state.trigger_noise_capture = True
        st.toast("Capturing new noise floor...")

# 5. WebRTC 스트리머
def video_transformer_factory():
    return OptimizedAudioProcessor(shared_queue)

ctx = webrtc_streamer(
    key="vocal-points-v1",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=video_transformer_factory,
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
)

if ctx.audio_processor and st.session_state.get("trigger_noise_capture"):
    ctx.audio_processor.capture_noise = True
    st.session_state.trigger_noise_capture = False

chart_placeholder = st.empty()

# 6. 메인 렌더링 루프
if ctx.state.playing:
    SAMPLING_RATE = 44100
    
    while ctx.state.playing:
        try:
            # 해상도가 실시간으로 바뀔 수 있으므로 루프 안에서 빈 생성
            freq_bins = np.linspace(0, target_hz, data_points)
            
            mag, noise_floor = shared_queue.get(timeout=0.03)
            
            if noise_floor is not None:
                final_mag = np.maximum(0, mag - noise_floor)
            else:
                final_mag = mag
            
            freqs = np.fft.fftfreq(len(mag)*2, 1/SAMPLING_RATE)[:len(mag)]
            indices = np.searchsorted(freqs, freq_bins)
            indices = np.clip(indices, 0, len(final_mag) - 1)
            y_data = final_mag[indices]
            
            y_clipped = np.clip(y_data, 0, y_max_val)
            chart_df = pd.DataFrame({
                "Voice": y_clipped,
                "Limit": y_max_val
            }, index=freq_bins.astype(int))
            
            chart_placeholder.line_chart(
                chart_df, 
                height=500, 
                width='stretch'
            )
            
        except queue.Empty:
            continue
        except Exception:
            continue
else:
    st.info("🎤 Press 'Start' to begin. Adjust 'Data Points' for higher resolution.")