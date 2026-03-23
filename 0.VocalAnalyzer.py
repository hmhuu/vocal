#pip install numpy matplotlib scipy pyaudio
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
from scipy.interpolate import make_interp_spline

# Matplotlib 기본 단축키 해제 (s키 충돌 방지)
plt.rcParams['keymap.save'] = '' 

# ==========================================
# [저장된 설정값 적용]
# ==========================================
DATA_POINTS = 100         
MAX_INTENSITY = 100000    
TARGET_HZ = 3000          
SMOOTH_FACTOR = 300       
# ==========================================

CHUNK = 2048                
RATE = 44100                
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

is_paused = False
noise_floor = np.zeros(DATA_POINTS)
harmony_lines = []

def note_to_hz(note):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    try:
        name = note[:-1].upper().replace('DB', 'C#').replace('EB', 'D#').replace('GB', 'F#').replace('AB', 'G#').replace('BB', 'A#')
        octave = int(note[-1])
        n = notes.index(name)
        return 440.0 * (2.0 ** ((n - 9 + (octave - 4) * 12) / 12.0))
    except: return None

fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.25)

x_raw = np.linspace(0, TARGET_HZ, DATA_POINTS)
x_smooth = np.linspace(0, TARGET_HZ, SMOOTH_FACTOR)
line, = ax.plot(x_smooth, np.zeros(SMOOTH_FACTOR), color='dodgerblue', lw=2, zorder=3)

# --- 마우스 추적용 가이드라인 및 텍스트 (다시 추가) ---
v_line = ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, visible=False, zorder=2)
cursor_text = ax.text(0, 0, '', fontsize=10, color='red', fontweight='bold',
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round'),
                      zorder=5)

ax.set_xlim(0, TARGET_HZ)
ax.set_ylim(0, MAX_INTENSITY)
ax.set_title("Vocal Analyzer (Mouse Tracking & Noise Reduction)")
ax.grid(axis='y', linestyle='--', alpha=0.2)

# --- 마우스 이동 이벤트 핸들러 ---
def on_mouse_move(event):
    if event.inaxes == ax:
        v_line.set_xdata([event.xdata])
        v_line.set_visible(True)
        # 커서 위치에 따라 텍스트 위치 조정
        cursor_text.set_position((event.xdata + 50, event.ydata + (MAX_INTENSITY * 0.05)))
        cursor_text.set_text(f"{int(event.xdata)} Hz\n{int(event.ydata)}")
        cursor_text.set_visible(True)
    else:
        v_line.set_visible(False)
        cursor_text.set_visible(False)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# --- 잡음 학습 로직 ---
def set_noise_action(event):
    global noise_floor
    print("잡음 수집 중...")
    samples = []
    for _ in range(10):
        data = stream.read(CHUNK, exception_on_overflow=False)
        mag = np.abs(np.fft.rfft(np.frombuffer(data, dtype=np.int16)))
        freqs = np.fft.rfftfreq(CHUNK, 1/RATE)
        indices = np.digitize(freqs, np.linspace(0, TARGET_HZ, DATA_POINTS + 1))
        tmp = np.zeros(DATA_POINTS)
        for i in range(DATA_POINTS):
            mask = (indices == i + 1)
            if np.any(mask): tmp[i] = np.mean(mag[mask])
        samples.append(tmp)
    noise_floor = np.mean(samples, axis=0)
    print("잡음 제거 완료!")

# --- 화성 가이드 ---
def draw_harmony(event):
    global harmony_lines
    for l in harmony_lines: l.remove()
    harmony_lines = []
    base_hz = note_to_hz(text_box.text.strip())
    if base_hz and base_hz < TARGET_HZ:
        for i in range(1, 10):
            freq = base_hz * i
            if freq > TARGET_HZ: break
            l = ax.axvline(x=freq, color='forestgreen' if i==1 else 'forestgreen', 
                           alpha=0.6 if i==1 else 0.2, lw=1.5, zorder=1)
            harmony_lines.append(l)
    fig.canvas.draw_idle()

# UI 구성
ax_box = plt.axes([0.15, 0.05, 0.1, 0.05])
text_box = TextBox(ax_box, 'Note: ', initial='C4')
ax_harm = plt.axes([0.27, 0.05, 0.1, 0.05])
btn_harm = Button(ax_harm, 'Harmony', color='honeydew')
ax_noise = plt.axes([0.45, 0.05, 0.1, 0.05])
btn_noise = Button(ax_noise, 'Set Noise', color='gold')
ax_pause = plt.axes([0.6, 0.05, 0.08, 0.05])
btn_pause = Button(ax_pause, 'Stop(s)', color='lightcoral')
ax_resume = plt.axes([0.7, 0.05, 0.08, 0.05])
btn_resume = Button(ax_resume, 'Run(r)', color='lightgreen')

btn_harm.on_clicked(draw_harmony)
btn_noise.on_clicked(set_noise_action)

def on_key(event):
    global is_paused
    if event.key == 's': is_paused = True
    elif event.key == 'r': is_paused = False

fig.canvas.mpl_connect('key_press_event', on_key)
btn_pause.on_clicked(lambda e: on_key(type('obj', (object,), {'key': 's'})))
btn_resume.on_clicked(lambda e: on_key(type('obj', (object,), {'key': 'r'})))

def update(frame):
    if is_paused: return [line, v_line, cursor_text] + harmony_lines
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        fft_mag = np.abs(np.fft.rfft(np.frombuffer(data, dtype=np.int16)))
        freqs = np.fft.rfftfreq(CHUNK, 1/RATE)
        indices = np.digitize(freqs, np.linspace(0, TARGET_HZ, DATA_POINTS + 1))
        y_raw = np.zeros(DATA_POINTS)
        for i in range(DATA_POINTS):
            mask = (indices == i + 1)
            if np.any(mask):
                val = np.mean(fft_mag[mask])
                y_raw[i] = max(0, val - noise_floor[i] * 1.1)
        spline = make_interp_spline(x_raw, y_raw, k=3)
        line.set_ydata(np.maximum(0, spline(x_smooth)))
    except: pass
    return [line, v_line, cursor_text] + harmony_lines

ani = FuncAnimation(fig, update, blit=False, interval=30, cache_frame_data=False)
plt.show()
stream.stop_stream(); stream.close(); p.terminate()