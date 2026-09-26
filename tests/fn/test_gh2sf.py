"""gh2sf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gh2sf import gh2sf


def test_gh2sf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gh2sf()
