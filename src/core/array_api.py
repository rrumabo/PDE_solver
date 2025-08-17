# minimal array backend
def get_xp(use_gpu: bool = False):
    if use_gpu:
        try:
            import cupy as cp
            return cp
        except Exception:
            pass
    import numpy as np
    return np

def to_numpy(a):
    import numpy as np
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)