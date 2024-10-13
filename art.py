import numpy as np


def getColors(n):
    m = np.ceil(n ** (1 / 3))
    m2 = m * m

    def split(x):
        return x // m2, (x % m2) // m, x % m

    lst = [split(i) for i in range(n)]
    arr = (0.5 + 0.45 * (np.array(lst) / m - 0.5) * 2) * 255
    arr = np.round(arr).astype(np.uint8)
    return arr.tolist()
