"""svipm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svipm import ideal_point_mle


def test_svipm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ideal_point_mle(data=None)
