import numpy as np
import nrrd
import matplotlib.pyplot as plt
import os
import math

file = os.path.join('..', '..', '..', 'Daten2', 'Fibroblasten', '3_draq5.nrrd')
data, header = nrrd.read(file)
data = data[:,:,0:614]
hist = []
for i in range(data.shape[2]):
    plane = data[:,:,i]
    mean_brightness = np.mean(plane[plane>10])
    hist.append(mean_brightness)
hist = np.array(hist)
for i in range(len(hist)):
    if math.isnan(hist[i]):
        hist[i] = 0
x = np.linspace(0, len(hist), len(hist)).astype(np.int)
fig = plt.figure(figsize=(5,6))
ax = plt.axes()
ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
ax.xaxis.tick_top()                     # and move the X-Axis      
ax.yaxis.set_ticks(np.arange(0, len(hist), 50)) # set y-ticks
ax.yaxis.tick_left()                    # remove right y-Ticks
ax.plot(hist, x)
plt.xlabel('Mittlere Helligkeit von Voxelwerten > 10')
plt.ylabel('Slice-Nr.')
plt.savefig('Helligkeitsgradient.svg', format='svg')
