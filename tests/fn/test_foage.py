"""foage is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.foage import foage


def test_foage_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        foage()
