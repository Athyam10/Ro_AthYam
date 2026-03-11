import serial
import matplotlib.pyplot as plt
import re

# CHANGE THIS TO YOUR ARDUINO PORT
ser = serial.Serial('COM9', 115200)

sensor_values = [0,0,0,0,0,0]

plt.ion()
fig, ax = plt.subplots()

bars = ax.bar(["S1","S2","S3","S4","S5","S6"], sensor_values)

ax.set_ylim(0, 400)
ax.set_ylabel("Distance (cm)")
ax.set_title("Ultrasonic Sensor Distances")

while True:

    line = ser.readline().decode('utf-8').strip()

    numbers = re.findall(r'\d+', line)

    if len(numbers) >= 6:
        sensor_values = list(map(int, numbers[:6]))

        for bar, val in zip(bars, sensor_values):
            bar.set_height(val)

        plt.pause(0.01)