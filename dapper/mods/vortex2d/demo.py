"""Demonstrate the 2D vortex model."""

import numpy as np
from matplotlib import pyplot as plt

import dapper.mods.vortex2d as vortex2d
from dapper.tools.progressbar import progbar
import dapper.tools.viz as viz

def show(x, model, ax=None):
    u, v, q, p = model.expand(x)
    wind = vortex2d.uv2wspd(u, v)
    im = ax.imshow(wind, cmap='plasma')
    def update(x):
        u, v, q, p = model.expand(x)
        wind = vortex2d.uv2wspd(u, v)
        im.set_data(wind)
        plt.pause(0.01)
    return update

fig = plt.figure(figsize=(8, 4))
ax = plt.subplot(111)
ax.set_aspect('equal', viz.adjustable_box_or_forced())
ax.set_title('Wind speed (m/s)')

xx = np.load(vortex2d.sample_filename)['sample']
setter = show(xx[0], vortex2d.model_config(), ax=ax)
for k, x in progbar(list(enumerate(xx)), "Animating"):
    if k%1 == 0:
        setter(x)
        fig.suptitle("time step: "+str(k))
        plt.pause(0.01)
