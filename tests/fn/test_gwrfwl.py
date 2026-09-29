"""Tests for morie.fn.gwrfwl: the FWL coefficients equal the full local regression block."""

import math

from morie.fn import _array_core as np
from morie.fn.gwrfwl import gwrfwl

N = 16
P = [(float(i % 4), float(i // 4)) for i in range(N)]
X1 = [[P[i][0]] for i in range(N)]
X2 = [[(0.3 * i) % 1.7, math.cos(i)] for i in range(N)]
Y = [1.0 + 2.0 * X2[i][0] - X2[i][1] + 0.1 * P[i][0] + 0.2 * math.sin(i) for i in range(N)]


def test_fwl_identity():
    r = gwrfwl(Y, X1, X2, P, 2.5, kernel="gaussian")
    for i in (0, 9):
        w = [math.exp(-(math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) ** 2) / (2 * 2.5**2)) for j in range(N)]
        Z = np.array([[1.0, X1[j][0], X2[j][0], X2[j][1]] for j in range(N)])
        Wd = np.diag(np.array(w))
        b = np.linalg.inv(Z.T @ Wd @ Z) @ (Z.T @ Wd @ np.array(Y))
        assert abs(r["betas"][i][0] - float(b[2])) < 1e-9
        assert abs(r["betas"][i][1] - float(b[3])) < 1e-9
