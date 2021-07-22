# gaussian_filter

Use this Python code to compare SciPy speed to our benchmark. Yes, it's varying wildly.

    from datetime import datetime
    import numpy as np
    from scipy.ndimage.filters import gaussian_filter

    # Warp up
    a = np.zeros((134, 156, 130), dtype='float')
    g = gaussian_filter(a, 1.0, truncate=2)

    # Time
    start = datetime.now()
    a = np.zeros((134, 156, 130), dtype='float')
    g = gaussian_filter(a, 1.0, truncate=2)
    print((datetime.now() - start).microseconds / 1000, 'ms')
    
