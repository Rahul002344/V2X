# adaptive_signal.py
def compute_green_time(density, base_green=10, max_green=60):
    """
    Simple mapping: density in [0,1] -> green time in seconds.
    base_green: minimum green seconds.
    max_green: maximum allowed green seconds.
    """
    d = max(0.0, min(1.0, float(density)))
    return base_green + d * (max_green - base_green)

def proportional_green(densities, cycle_time=120, min_green=5):
    """
    densities: list/array of densities per arm (0..1)
    cycle_time: total cycle time in seconds
    Returns list of green times proportional to densities, but respecting min_green.
    """
    import numpy as np
    arr = np.array(densities, dtype=float)
    if arr.sum() == 0:
        return [cycle_time/len(arr)] * len(arr)
    raw = (arr / arr.sum()) * cycle_time
    trimmed = [max(min_green, r) for r in raw]
    # if trimmed sum > cycle_time, normalize down
    s = sum(trimmed)
    if s > cycle_time:
        factor = cycle_time / s
        trimmed = [r * factor for r in trimmed]
    return trimmed