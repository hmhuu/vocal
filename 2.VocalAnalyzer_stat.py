#pip install psutil
#pip install numpy matplotlib scipy pyaudio
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
from scipy.interpolate import make_interp_spline
import psutil
import os

# Matplotlib 기본 설정 및 툴바 제거
plt.rcParams['keymap.save'] = ''
plt.rcParams['toolbar'] = 'None'

# ==========================================
# [초기 설정값]
# ==========================================
data_points = 80           
DEFAULT_INTENSITY = 100000    
DEFAULT_HZ = 3000          
SMOOTH_FACTOR = 150        
CHUNK = 2048                
RATE = 44100                

# 전역 상태 변수
target_hz = DEFAULT_HZ
is_paused = False
noise_floor = np.zeros(data_points)
harmony_lines = []
process = psutil.Process(os.getpid())

# 데이터 계산용 배열
x_raw = np.linspace(0, target_hz, data_points)
x_smooth = np.linspace(0, target_hz, SMOOTH_FACTOR)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# --- 메인 그래프 UI 생성 ---
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2, top=0.9) # 상단/하단 여백 확보

line, = ax.plot(x_smooth, np.zeros(SMOOTH_FACTOR), color='dodgerblue', lw=2, zorder=3, animated=True)
v_line = ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, visible=False, zorder=2, animated=True)
cursor_text = ax.text(0, 0, '', fontsize=9, color='red', fontweight='bold',
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round'),
                      zorder=5, animated=True)

# 리소스 정보 (하단 버튼 라인 우측 끝)
resource_text = fig.text(0.98, 0.05, '', fontsize=9, color='dimgray', 
                         ha='right', va='center', fontweight='bold')

ax.set_xlim(0, target_hz)
ax.set_ylim(0, DEFAULT_INTENSITY)
ax.set_title("Vocal Analyzer", pad=20)
ax.grid(axis='y', linestyle='--', alpha=0.2)

# --- 로직 함수들 ---
def note_to_hz(note):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    try:
        name = note[:-1].upper().replace('DB', 'C#').replace('EB', 'D#').replace('GB', 'F#').replace('AB', 'G#').replace('BB', 'A#')
        octave = int(note[-1])
        n = notes.index(name)
        return 440.0 * (2.0 ** ((n - 9 + (octave - 4) * 12) / 12.0))
    except: return None

def draw_harmony(event):
    global harmony_lines
    for l in harmony_lines:
        try: l.remove()
        except: pass
    harmony_lines = []
    base_hz = text_box.text.strip()
    hz = note_to_hz(base_hz)
    if hz and hz < target_hz:
        for i in range(1, 8):
            freq = hz * i
            if freq > target_hz: break
            l = ax.axvline(x=freq, color='forestgreen', alpha=0.5 if i==1 else 0.15, lw=1, zorder=1, animated=True)
            harmony_lines.append(l)
    fig.canvas.draw_idle()

def set_noise_action(event):
    global noise_floor
    samples = []
    for _ in range(10):
        data = stream.read(CHUNK, exception_on_overflow=False)
        mag = np.abs(np.fft.rfft(np.frombuffer(data, dtype=np.int16)))
        freqs = np.fft.rfftfreq(CHUNK, 1/RATE)
        indices = np.digitize(freqs, np.linspace(0, target_hz, data_points + 1))
        tmp = np.zeros(data_points)
        for i in range(data_points):
            mask = (indices == i + 1)
            if np.any(mask): tmp[i] = np.mean(mag[mask])
        samples.append(tmp)
    noise_floor = np.mean(samples, axis=0)

# --- 설정창 관리 클래스 ---
class ConfigPopup:
    def __init__(self, main_fig, main_ax):
        self.fig = None
        self.main_fig = main_fig
        self.main_ax = main_ax

    def open(self, event):
        if self.fig and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        self.fig = plt.figure("Settings", figsize=(5, 3.5))
        ax_int = plt.axes([0.45, 0.70, 0.4, 0.12])
        self.txt_int = TextBox(ax_int, 'Intensity MAX: ', initial=str(int(self.main_ax.get_ylim()[1])))
        ax_hz = plt.axes([0.45, 0.50, 0.4, 0.12])
        self.txt_hz = TextBox(ax_hz, 'Hz MAX: ', initial=str(int(self.main_ax.get_xlim()[1])))
        ax_pts = plt.axes([0.45, 0.30, 0.4, 0.12])
        self.txt_pts = TextBox(ax_pts, 'Points (Max 300): ', initial=str(data_points))
        ax_ok = plt.axes([0.7, 0.05, 0.2, 0.12])
        self.btn_ok = Button(ax_ok, 'OK', color='lightgray')
        self.btn_ok.on_clicked(self.apply)
        plt.show()

    def apply(self, event):
        global target_hz, data_points, x_raw, x_smooth, noise_floor
        try:
            target_hz = float(self.txt_hz.text)
            data_points = min(int(float(self.txt_pts.text)), 300)
            self.main_ax.set_ylim(0, float(self.txt_int.text))
            self.main_ax.set_xlim(0, target_hz)
            x_raw = np.linspace(0, target_hz, data_points)
            x_smooth = np.linspace(0, target_hz, SMOOTH_FACTOR)
            noise_floor = np.zeros(data_points)
            self.main_fig.canvas.draw() 
            plt.close(self.fig)
        except ValueError: pass

config_handler = ConfigPopup(fig, ax)

# --- 버튼 구성 (Config 버튼은 우측 상단으로) ---
# [변경] 우측 상단 위치: [left, bottom, width, height]
ax_config = plt.axes([0.85, 0.91, 0.1, 0.04]) 
btn_config = Button(ax_config, 'Config', color='whitesmoke')
btn_config.on_clicked(config_handler.open)

# 하단 버튼들
ax_box = plt.axes([0.05, 0.05, 0.07, 0.05])
text_box = TextBox(ax_box, 'Note: ', initial='C4')

ax_harm = plt.axes([0.13, 0.05, 0.09, 0.05])
btn_harm = Button(ax_harm, 'Harmony', color='honeydew')
btn_harm.on_clicked(draw_harmony)

ax_noise = plt.axes([0.25, 0.05, 0.09, 0.05])
btn_noise = Button(ax_noise, 'Set Noise', color='gold')
btn_noise.on_clicked(set_noise_action)

ax_pause = plt.axes([0.37, 0.05, 0.08, 0.05])
btn_pause = Button(ax_pause, 'Stop(s)', color='lightcoral')

ax_resume = plt.axes([0.47, 0.05, 0.08, 0.05])
btn_resume = Button(ax_resume, 'Run(r)', color='lightgreen')

# --- 핸들러 및 애니메이션 ---
def on_mouse_move(event):
    if event.inaxes == ax:
        v_line.set_xdata([event.xdata])
        v_line.set_visible(True)
        cursor_text.set_position((event.xdata + (target_hz * 0.02), ax.get_ylim()[1] * 0.05 + event.ydata))
        cursor_text.set_text(f"{int(event.xdata)} Hz\n{int(event.ydata)}")
        cursor_text.set_visible(True)
    else:
        v_line.set_visible(False)
        cursor_text.set_visible(False)

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

def on_key(event):
    global is_paused
    if event.key == 's': is_paused = True
    elif event.key == 'r': is_paused = False

fig.canvas.mpl_connect('key_press_event', on_key)
btn_pause.on_clicked(lambda e: on_key(type('obj', (object,), {'key': 's'})))
btn_resume.on_clicked(lambda e: on_key(type('obj', (object,), {'key': 'r'})))

def update(frame):
    cpu_usage = psutil.cpu_percent()
    ram_usage = process.memory_info().rss / 1024 / 1024
    resource_text.set_text(f"CPU: {cpu_usage}%  |  RAM: {ram_usage:.1f}MB")

    artists = [line, v_line, cursor_text] + harmony_lines
    if is_paused: return artists

    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        fft_mag = np.abs(np.fft.rfft(np.frombuffer(data, dtype=np.int16)))
        freqs = np.fft.rfftfreq(CHUNK, 1/RATE)
        indices = np.digitize(freqs, np.linspace(0, target_hz, data_points + 1))
        y_raw = np.zeros(data_points)
        for i in range(data_points):
            mask = (indices == i + 1)
            if np.any(mask):
                val = np.mean(fft_mag[mask])
                y_raw[i] = max(0, val - noise_floor[i] * 1.1)
        spline = make_interp_spline(x_raw, y_raw, k=3)
        line.set_ydata(np.maximum(0, spline(x_smooth)))
    except: pass
    return artists

ani = FuncAnimation(fig, update, blit=True, interval=30, cache_frame_data=False)
plt.show()

stream.stop_stream(); stream.close(); p.terminate()