"""gdrel is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdrel import gdrel


def test_gdrel_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdrel()
