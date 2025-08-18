import numpy as np

def gaussian_bump(x, center=0.0, sigma=0.5, amp=1.0):
    return amp * np.exp(-((x - center) / sigma)**2)