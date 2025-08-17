import numpy as np

def gaussian_bump(x, center=0.0, width=0.5, amplitude=1.0):
    return amplitude * np.exp(-((x - center) / width)**2)