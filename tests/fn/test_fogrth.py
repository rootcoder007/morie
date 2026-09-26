"""fogrth is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fogrth import fogrth


def test_fogrth_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fogrth()
