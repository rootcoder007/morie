"""svipe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svipe import ideal_point_em


def test_svipe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ideal_point_em(data=None)
