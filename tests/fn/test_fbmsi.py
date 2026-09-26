"""fbmsi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fbmsi import fbmsi


def test_fbmsi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fbmsi()
