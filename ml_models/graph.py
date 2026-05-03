import pandas as pd
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ----------------
df = pd.read_csv("traffic_analysis.csv")

# ---------------- 1. DENSITY GRAPH ----------------
plt.figure()
plt.plot(df["frame"], df["north"], label="North")
plt.plot(df["frame"], df["south"], label="South")
plt.plot(df["frame"], df["east"], label="East")
plt.plot(df["frame"], df["west"], label="West")

plt.xlabel("Frame")
plt.ylabel("Density")
plt.title("Traffic Density per Direction")
plt.legend()
plt.grid()

plt.savefig("density_graph.png")
plt.show()

# ---------------- 2. SIGNAL TIMING GRAPH ----------------
# Convert phase into numeric
df["NS_active"] = df["phase"].apply(lambda x: 1 if x=="NS" else 0)
df["EW_active"] = df["phase"].apply(lambda x: 1 if x=="EW" else 0)

plt.figure()
plt.plot(df["frame"], df["NS_active"], label="NS Phase")
plt.plot(df["frame"], df["EW_active"], label="EW Phase")

plt.xlabel("Frame")
plt.ylabel("Signal State")
plt.title("Signal Phase Over Time")
plt.legend()
plt.grid()

plt.savefig("signal_phase_graph.png")
plt.show()

# ---------------- 3. TIME REMAINING GRAPH ----------------
plt.figure()
plt.plot(df["frame"], df["time"], label="Remaining Time")

plt.xlabel("Frame")
plt.ylabel("Time (seconds)")
plt.title("Signal Countdown Timer")
plt.legend()
plt.grid()

plt.savefig("timer_graph.png")
plt.show()

# ---------------- 4. PHASE DISTRIBUTION ----------------
phase_counts = df["phase"].value_counts()

plt.figure()
phase_counts.plot(kind='bar')

plt.title("Signal Phase Distribution")
plt.xlabel("Phase")
plt.ylabel("Count")

plt.savefig("phase_distribution.png")
plt.show()