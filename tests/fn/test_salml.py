"""salml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.salml import salml


def test_salml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        salml()
