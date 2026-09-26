"""ddpmst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ddpmst import ddpm_step


def test_ddpmst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ddpm_step(x_t=None, t=None, eps_theta=None)
