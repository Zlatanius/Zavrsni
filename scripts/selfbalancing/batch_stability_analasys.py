import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import glob

def analyze_run(file_path):
    data = np.load(file_path)
    time = data["time"][:-2]
    angle = data["tilt"][:-2]

    # --- Filter ---
    fs = 1 / np.mean(np.diff(time))
    cutoff = 10.0
    order = 2
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    angle = filtfilt(b, a, angle)

    if angle[0] > 0:
        angle = -angle  # invertuj ugao ako je negativan na početku

    # >>> Skip ako je početni ugao < 3°
    # if abs(angle[0]) < 1.0:
    #     return None

    # --- Derivacija ---
    dt = np.mean(np.diff(time))
    derivative = np.gradient(angle, dt)

    # Overshoot = prvo lokalno maksimum
    overshoot_idx = None
    for i in range(1, len(derivative)):
        if derivative[i-1] > 0 and derivative[i] <= 0:
            overshoot_idx = i
            break
    if overshoot_idx is not None:
        overshoot = angle[overshoot_idx]
    else:
        overshoot = 0.0

    # Steady-state
    steady_state_value = np.mean(angle[int(0.9 * len(angle)):])

    # Settling time
    settling_threshold = 0.05 * abs(overshoot)
    settling_time = None
    for i in range(len(angle)):
        if np.all(np.abs(angle[i:]) < settling_threshold):
            settling_time = time[i]
            break

    # Time in balance
    balance_threshold = 0.1 * abs(overshoot)
    time_in_balance = np.sum(np.abs(angle) < balance_threshold) * dt

    # Oscilacije
    zero_crossings = np.where(np.diff(np.sign(angle)))[0]
    num_oscillations = len(zero_crossings)

    return {
        "overshoot": overshoot,
        "steady_state": steady_state_value,
        "settling_time": settling_time if settling_time else np.nan,
        "time_in_balance": time_in_balance,
        "oscillations": num_oscillations,
    }


# === Učitaj sve fajlove ===
files = glob.glob("D:/isaaclab/tilt_logging/tilt_log_run*.npz")
print(f"Nađeno fajlova: {len(files)}")

results = [analyze_run(f) for f in files]
small_angle = [r for r in results if r is None]
results = [r for r in results if r is not None]  # filtriraj None

print(f"Broj korištenih run-ova: {len(results)}")
print (f"Preskočeno zbog malog početnog ugla: {len(small_angle)}")

# === Pretvori u numpy za lakše crtanje ===
overshoots = np.array([r["overshoot"] for r in results])
steady_states = np.array([r["steady_state"] for r in results])
settling_times = np.array([r["settling_time"] for r in results])
times_in_balance = np.array([r["time_in_balance"] for r in results])
oscillations = np.array([r["oscillations"] for r in results])

# === Histogrami / distribucije ===
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.hist(overshoots, bins=10, color="orange", edgecolor="black")
plt.title("Distribucija overshoot-a")
plt.xlabel("Overshoot [°]")

plt.subplot(2, 3, 2)
plt.hist(steady_states, bins=10, color="purple", edgecolor="black")
plt.title("Distribucija steady-state vrijednosti")
plt.xlabel("Steady-state [°]")

plt.subplot(2, 3, 3)
plt.hist(settling_times[~np.isnan(settling_times)], bins=10, color="black", edgecolor="black")
plt.title("Distribucija vremena smirivanja")
plt.xlabel("T_s [s]")

plt.subplot(2, 3, 4)
plt.hist(times_in_balance, bins=10, color="blue", edgecolor="black")
plt.title("Distribucija vremena u balansu")
plt.xlabel("Vrijeme [s]")

plt.subplot(2, 3, 5)
plt.hist(oscillations, bins=np.arange(0, max(oscillations)+2)-0.5, color="green", edgecolor="black")
plt.title("Distribucija broja oscilacija")
plt.xlabel("Broj oscilacija")

plt.tight_layout()
plt.show()
