"""Demonstrate the 2D vortex model."""

import numpy as np
from matplotlib import pyplot as plt

import dapper.mods.vortex2d as vortex2d
from dapper.tools.progressbar import progbar
import dapper.tools.viz as viz

def show(x, model, ax=None):
    u, v, q, p = model.expand(x)
    zeta = vortex2d.uv2zeta(u, v, model.dx)
    im = ax.imshow(zeta)
    def update(x):
        u, v, q, p = model.expand(x)
        zeta = vortex2d.uv2zeta(u, v, model.dx)
        im.set_data(zeta)
        plt.pause(0.01)
    return update

xx = np.load(vortex2d.sample_filename)['sample']

fig = plt.figure(figsize=(8, 4))
ax = plt.subplot(111)
ax.set_aspect('equal', viz.adjustable_box_or_forced())
ax.set_title('')

for k, x in progbar(list(enumerate(xx)), "Animating"):
    if k%1 == 0:
        show(x, vortex2d.model_config(), ax=ax)
        fig.suptitle("k: "+str(k))
        plt.pause(0.01)
