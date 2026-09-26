"""svipa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svipa import ideal_point_adapt


def test_svipa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ideal_point_adapt(data=None)
