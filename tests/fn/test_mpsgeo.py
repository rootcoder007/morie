"""mpsgeo: hard data honoured, values drawn from the training image, patterns reproduced."""

import math

from morie.fn.mpsgeo import direct_sampling

TI = [[1.0 if (i // 3 + j // 3) % 2 == 0 else 0.0 for j in range(18)] for i in range(18)]


def test_direct_sampling_properties():
    cond = [(0, 0, 1.0), (4, 5, 0.0)]
    r = direct_sampling(TI, 9, 9, n_neighbors=8, threshold=0.0, max_fraction=1.0, conditioning=cond, seed=2)
    assert r.grid[0][0] == 1.0 and r.grid[4][5] == 0.0
    assert all(v in (0.0, 1.0) for row in r.grid for v in row)
    assert len(r.path) == 81 - 2
    again = direct_sampling(TI, 9, 9, n_neighbors=8, threshold=0.0, max_fraction=1.0, conditioning=cond, seed=2)
    assert again.grid == r.grid
    cont = [[math.sin(i / 3) + j / 10 for j in range(10)] for i in range(10)]
    c = direct_sampling(cont, 5, 5, categorical=False, seed=1)
    vals = {v for row in cont for v in row}
    assert all(v in vals for row in c.grid for v in row)
