"""A 2D vorticity equation model to describe vortex motion.
    model state is defined on square domain with 2^x grid points
    reads in u,v field as model state
    the default state is a rankine vortex embedded in random background flow
    author: Yue Ying, 2021
"""
import numpy as np

import dapper.mods as modelling


class model_config:

    dx       = 9000  ##meter
    nx       = 64
    dt       = 300   ##second
    dtout    = 3600
    Rmw      = 10
    Vmax     = 20
    gen_rate = 2.0
    diss     = 5e3

    def __init__(self):
        self.M = self.nx * self.nx * 3 + 2

    def expand(self, x):
        n = self.nx
        u = np.reshape(x[0:n*n], (n, n))
        v = np.reshape(x[n*n:2*n*n], (n, n))
        q = np.reshape(x[2*n*n:3*n*n], (n, n))
        p = x[3*n*n:]
        return u, v, q, p

    def vectorize(self, u, v, q, p):
        x = np.zeros((self.M))
        n = self.nx
        x[0:n*n] = np.reshape(u, (n*n,))
        x[n*n:2*n*n] = np.reshape(v, (n*n,))
        x[2*n*n:3*n*n] = np.reshape(q, (n*n,))
        x[3*n*n:] = p
        return x

    def initial_condition(self):
        ##background flow:
        power_law = -3
        pk = lambda k: k**((power_law-1)/2)
        zeta = 1e-4 * gaussian_random_field(pk, self.nx)
        u0, v0 = zeta2uv(zeta, self.dx)
        ##add in the rankine vortex
        iStorm, jStorm = (self.nx/2, self.nx/2)
        u, v = rankine_vortex(self.nx, iStorm, jStorm, self.Rmw, self.Vmax)
        ##tracor
        q = np.zeros((self.nx, self.nx))
        ##storm location stored in p
        p = np.array([iStorm, jStorm])
        x = self.vectorize(u+u0, v+v0, q, p)
        return x

    def step(self, x, t, dtout):
        u, v, q, p = self.expand(x)
        zeta = uv2zeta(u, v, self.dx)
        diss = 5e3
        gen = self.gen_rate*1e-5*wind_cutoff(np.max(uv2wspd(u, v)), 70)
        for n in range(int(dtout/self.dt)):
            rhs1 = forcing(u, v, zeta, diss, gen, self.dx)
            zeta1 = zeta + 0.5*self.dt*rhs1
            rhs2 = forcing(u, v, zeta1, diss, gen, self.dx)
            zeta2 = zeta + 0.5*self.dt*rhs2
            rhs3 = forcing(u, v, zeta2, diss, gen, self.dx)
            zeta3 = zeta + self.dt*rhs3
            rhs4 = forcing(u, v, zeta3, diss, gen, self.dx)
            zeta = zeta + self.dt*(rhs1/6.0+rhs2/3.0+rhs3/3.0+rhs4/6.0)
            u, v = zeta2uv(zeta, self.dx)
        x = self.vectorize(u, v, q, p)
        return x

def deriv_x(f, dx):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))/(2.0*dx)

def deriv_y(f, dx):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))/(2.0*dx)

def laplacian(f, dx):
    return ((np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) + np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1)) - 4.0*f)/(dx**2)

def wind_cutoff(wind, max_wind):
  buff = 10.0
  f = 0.0
  if (wind < max_wind-buff):
    f = 1.0
  if (wind >= max_wind-buff and wind < max_wind):
    f = (max_wind - wind) / buff
  return f

def forcing(u, v, zeta, diss, gen, dx):
  fzeta = -(u*deriv_x(zeta, dx)+v*deriv_y(zeta, dx)) + gen*zeta + diss*laplacian(zeta, dx)
  return fzeta

def psi2uv(psi, dx):
  u = -(np.roll(psi, -1, axis=1) - psi)/dx
  v = (np.roll(psi, -1, axis=0) - psi)/dx
  return u, v

def uv2zeta(u, v, dx):
  zeta = (v - np.roll(v, 1, axis=0) - u + np.roll(u, 1, axis=1))/dx
  return zeta

def psi2zeta(psi, dx):
  zeta = ((np.roll(psi, -1, axis=0) + np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=1) + np.roll(psi, 1, axis=1)) - 4.0*psi)/(dx**2)
  return zeta

def zeta2uv(zeta, dx):
  psi = zeta2psi(zeta, dx)
  u, v = psi2uv(psi, dx)
  return u, v

def zeta2psi(zeta, dx):
  psi = np.zeros(zeta.shape)
  niter = 1000
  for i in range(niter):
    psi = ((np.roll(psi, -1, axis=0) + np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=1) + np.roll(psi, 1, axis=1)) - zeta*(dx**2))/4.0
  return psi

def uv2wspd(u, v):
  wspd = np.sqrt(u**2 + v**2)
  return wspd

def gaussian_random_field(pk, n):
    nup = int(n/2)
    wn = np.concatenate((np.arange(0, nup), np.arange(-nup, 0)))
    kx, ky = np.meshgrid(wn, wn)
    k2d = np.sqrt(kx**2 + ky**2)
    k2d[np.where(k2d==0.0)] = 1e-10
    noise = np.fft.fft2(np.random.normal(0, 1, (n, n)))
    amplitude = pk(k2d)
    amplitude[np.where(k2d==1e-10)] = 0.0
    noise1 = np.real(np.fft.ifft2(noise * amplitude))
    return (noise1 - np.mean(noise1))/np.std(noise1)

def rankine_vortex(nx, iStorm, jStorm, Rmw, Vmax):
    ii, jj = np.mgrid[0:nx, 0:nx]
    dist = np.sqrt((ii - iStorm)**2 + (jj - jStorm)**2)
    wspd = np.zeros((nx, nx))
    ind = np.where(dist <= Rmw)
    wspd[ind] = Vmax * dist[ind] / Rmw
    ind = np.where(dist > Rmw)
    wspd[ind] = Vmax * Rmw / dist[ind] * np.exp(-(dist[ind] - Rmw)**2/200)
    wspd[np.where(dist==0)] = 0
    dist[np.where(dist==0)] = 1e-10
    u = -wspd * (jj - jStorm) / dist
    v = wspd * (ii - iStorm) / dist
    return u, v

###generate inital samples
###
def gen_sample(model, nSamples, SpinUp, Spacing):
    simulator = modelling.with_recursion(model.step, prog="simulator")
    K = SpinUp + nSamples*Spacing
    sample = simulator(model.initial_condition(), K, 0.0, model.dtout)
    return sample[SpinUp::Spacing]

sample_filename = modelling.rc.dirs.samples/'vortex2d_samples.npz'
if not sample_filename.is_file():
    print('Did not find sample file', sample_filename, '. Generating...')
    sample = gen_sample(model_config(), 100, 0, 1)
    np.savez(sample_filename, sample=sample)
