import numpy as np

def gaussian_bump_2d(X, Y, x0=0.0, y0=0.0, sigma=0.5, amplitude=1.0):
    return amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

def gaussian_wave_packet(X, Y, amplitude=1.0, sigma=0.2, kx=0.0, ky=0.0):
    phase = np.exp(1j * (kx * X + ky * Y))
    envelope = amplitude * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return envelope * phase