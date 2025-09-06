import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Učitaj podatke iz .npz fajla
data = np.load("D:/isaaclab/tilt_logging/tilt_log_run19.npz")   # promijeni naziv fajla ako je drugačiji

time = data["time"][:-2]
angle = data["tilt"][:-2]

if angle[0] > 0:
    angle = -angle  # invertuj ugao ako je negativan na početku

# --- Parametri filtera ---
fs = 1 / np.mean(np.diff(time))  # frekvencija uzorkovanja
cutoff = 10.0  # granična frekvencija [Hz], niže = glađe
order = 2      # red filtera

# Kreiranje filtera
b, a = butter(order, cutoff/(0.5*fs), btype='low')
angle_smoothed = filtfilt(b, a, angle)
angle = angle_smoothed

# --- Izračun derivacije ---
dt = np.mean(np.diff(time))
derivative = np.gradient(angle, dt)

# Pronađi prvo mjesto gdje derivacija prelazi iz + u - (lokalni maksimum) ### CHANGED
overshoot_idx = None
for i in range(1, len(derivative)):
    if derivative[i-1] > 0 and derivative[i] <= 0:
        overshoot_idx = i
        break

if overshoot_idx is not None:
    overshoot = angle[overshoot_idx]
    overshoot_time = time[overshoot_idx]
else:
    overshoot = 0.0
    overshoot_time = None

# Početni ugao
start_angle = angle[0]

# --- Parametri za analizu ---
settling_threshold = 0.05 * abs(overshoot)   # prag za smirivanje (±5% overshoot-a)
balance_threshold = 0.1 * abs(overshoot)     # prag balansa (±10% overshoot-a)

# Steady-state vrijednost (prosjek zadnjih 10% vremena)
steady_state_value = np.mean(angle[int(0.9 * len(angle)):])

# Vrijeme smirivanja: prvo vrijeme nakon kojeg sve ostaje u granici ±settling_threshold
settling_time = None
for i in range(len(angle)):
    if np.all(np.abs(angle[i:]) < settling_threshold):
        settling_time = time[i]
        break

# Vrijeme u balansu: ukupno trajanje unutar ±balance_threshold
time_in_balance = np.sum(np.abs(angle) < balance_threshold) * dt

# Oscilacije – broj prelazaka kroz nulu
zero_crossings = np.where(np.diff(np.sign(angle)))[0]
num_oscillations = len(zero_crossings)

# --- Prikaz na grafu ---
plt.figure(figsize=(12, 6))
plt.plot(time, angle, label="Nagib robota", color="blue")
# plt.plot(time, angle_smoothed, label="Filtar", color="red", alpha=0.7)

# Nacrtaj granice
plt.axhline(settling_threshold, color="green", linestyle="--", label="Granica smirivanja")
plt.axhline(-settling_threshold, color="green", linestyle="--")

# Označi steady state
plt.axhline(
    steady_state_value,
    color="purple",
    linestyle=":",
    label=f"Steady-state = {steady_state_value:.2f}°"
)

# Označi overshoot tačku ### CHANGED
if overshoot_time:
    plt.axhline(
        overshoot,
        color="orange",
        linestyle="--",
        label=f"Overshoot = {overshoot:.2f}°"
    )
    plt.axvline(
        overshoot_time,
        color="orange",
        linestyle=":",
    )
    plt.scatter(overshoot_time, overshoot, color="orange", zorder=5)

# Dodaj liniju za vrijeme smirivanja (ako postoji)
if settling_time:
    plt.axvline(
        settling_time,
        color="black",
        linestyle="--",
        label=f"T_s = {settling_time:.2f} s"
    )

# Dodaj tekstualni summary na graf
textstr = '\n'.join((
    f"Početni ugao: {start_angle:.2f}°",
    f"Overshoot: {overshoot:.2f}°",
    f"Steady-state vrijednost: {steady_state_value:.2f}°",
    f"Vrijeme smirivanja: {settling_time:.2f} s" if settling_time else "Nije se smirio",
    f"Vrijeme u balansu (±{balance_threshold:.2f}°): {time_in_balance:.2f} s",
    f"Broj oscilacija: {num_oscillations}"
))
plt.gcf().text(0.65, 0.15, textstr, fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

# Završni detalji
plt.xlabel("Vrijeme [s]")
plt.ylabel("Nagib [°]")
plt.title("Analiza stabilnosti regulatora")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
