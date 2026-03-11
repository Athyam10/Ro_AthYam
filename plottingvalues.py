import serial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# -------- SERIAL PORT --------
# Change COM port depending on your system
ser = serial.Serial('COM3',115200,timeout=1)

# -------- SENSOR VARIABLES --------
S1=S2=S3=S4=S5=S6=0

# -------- RADAR ANGLES --------
angles = np.deg2rad([0,90,180,270])   # front,right,back,left

# -------- FIGURE SETUP --------
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(121, polar=True)
ax2 = fig.add_subplot(122)

# radar limits
ax1.set_ylim(0,200)

# bar graph
bars = ax2.bar(["Top (S1)","Bottom (S2)"],[0,0])
ax2.set_ylim(0,200)
ax2.set_ylabel("Distance (cm)")

# -------- UPDATE FUNCTION --------
def update(frame):

    global S1,S2,S3,S4,S5,S6

    try:
        line = ser.readline().decode().strip()

        if "S1" in line:

            parts=line.replace("cm","").replace(":","").split()

            S1=int(parts[1])
            S2=int(parts[3])
            S3=int(parts[5])
            S4=int(parts[7])
            S5=int(parts[9])
            S6=int(parts[11])

    except:
        return

    # -------- RADAR DATA --------
    radar_values=[S3,S4,S5,S6]

    ax1.clear()
    ax1.set_ylim(0,200)
    ax1.set_title("Rover Obstacle Radar")

    ax1.plot(angles,radar_values,'o-',linewidth=2)
    ax1.fill(angles,radar_values,alpha=0.25)

    ax1.set_thetagrids([0,90,180,270],
                       labels=["Front(S3)","Right(S4)","Back(S5)","Left(S6)"])

    # -------- BAR GRAPH --------
    bars[0].set_height(S1)
    bars[1].set_height(S2)

    ax2.set_title("Vertical Clearance")

# -------- ANIMATION --------
ani = FuncAnimation(fig,update,interval=100)

plt.tight_layout()
plt.show()