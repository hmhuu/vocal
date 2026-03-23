import streamlit as st
import numpy as np
from scipy.fftpack import fft
from scipy.interpolate import make_interp_spline
import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import queue
import logging
import time

# [1] 로그 차단: 터미널 경고 방지
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)

# 2. 페이지 설정
st.set_page_config(page_title="Vocal Analyzer Pro", layout="wide")
st.title("🎤 Advanced Vocal Analyzer Pro")

# 3. 세션 상태 초기화
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue(maxsize=15)
if "noise_floor" not in st.session_state:
    st.session_state.noise_floor = None 
if "prev_y" not in st.session_state:
    st.session_state.prev_y = None
if "last_df" not in st.session_state:
    st.session_state.last_df = None

shared_queue = st.session_state.audio_queue

# 4. 오디오 프로세서
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

# 5. 사이드바 설정
st.sidebar.header("⚙️ Settings")
y_max_val = st.sidebar.number_input("Intensity (Y-Max)", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
target_hz = st.sidebar.slider("Max Hz", 500, 5000, 3000)

st.sidebar.markdown("---")
data_points = st.sidebar.slider("Raw Data Points", 20, 150, 60)
smooth_factor = st.sidebar.slider("Smooth Factor (Spatial)", 50, 500, 200)
smoothing_alpha = st.sidebar.slider("Smoothing Alpha (Temporal)", 0.05, 1.0, 0.3)

# 6. 상단 컨트롤 레이아웃
c1, c2, c3 = st.columns([2, 1, 1])

with c1:
    ctx = webrtc_streamer(
        key="vocal-bugfix-v1",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=lambda: OptimizedAudioProcessor(shared_queue),
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )

with c2:
    st.write("") # 간격
    if st.button("🧹 Zero Noise", use_container_width=True):
        if ctx.audio_processor:
            ctx.audio_processor.capture_noise = True
            st.toast("Capturing noise floor...")

with c3:
    st.write("") # 간격
    # [버그 해결 핵심] st.button 대신 st.toggle 사용
    # 토글은 상태가 세션에 고정되므로 루프가 돌 때 반응이 즉각적입니다.
    graph_active = st.toggle("Graph Active", value=True)

chart_placeholder = st.empty()

# 7. 메인 분석 루프
if ctx.state.playing:
    SAMPLING_RATE = 44100
    
    while ctx.state.playing:
        try:
            # 큐에서 데이터를 꺼내는 작업은 항상 수행 (버퍼 방지)
            mag, noise_floor = shared_queue.get(timeout=0.03)
            
            # 토글 상태에 따라 업데이트 여부 결정
            if graph_active:
                x_raw = np.linspace(0, target_hz, data_points)
                x_smooth = np.linspace(0, target_hz, smooth_factor)
                
                if noise_floor is not None:
                    mag = np.maximum(0, mag - noise_floor)
                
                freqs = np.fft.fftfreq(len(mag)*2, 1/SAMPLING_RATE)[:len(mag)]
                indices = np.searchsorted(freqs, x_raw)
                indices = np.clip(indices, 0, len(mag) - 1)
                y_raw = mag[indices]
                
                # 시간적 감쇄
                if st.session_state.prev_y is None or len(st.session_state.prev_y) != data_points:
                    st.session_state.prev_y = y_raw
                y_temporal = (y_raw * smoothing_alpha) + (st.session_state.prev_y * (1 - smoothing_alpha))
                st.session_state.prev_y = y_temporal
                
                # 곡선 보간
                spline = make_interp_spline(x_raw, y_temporal, k=3)
                y_smooth = np.maximum(0, spline(x_smooth))
                
                # Y축 클리핑 및 데이터 저장
                y_clipped = np.clip(y_smooth, 0, y_max_val)
                st.session_state.last_df = pd.DataFrame({
                    "Voice": y_clipped,
                    "Limit": y_max_val
                }, index=x_smooth.astype(int))
            
            # 그래프 출력 (Active가 꺼져있으면 마지막 last_df를 계속 렌더링)
            if st.session_state.last_df is not None:
                chart_placeholder.line_chart(
                    st.session_state.last_df, 
                    height=500, 
                    width='stretch'
                )
            
            # CPU 점유율 조절 및 이벤트 큐 처리 보조
            time.sleep(0.001)
            
        except queue.Empty:
            continue
        except Exception:
            continue
else:
    if st.session_state.last_df is not None:
        chart_placeholder.line_chart(st.session_state.last_df, height=500, width='stretch')
    st.info("🎤 Start를 눌러주세요. Graph Active 토글로 실시간 분석을 멈출 수 있습니다.")