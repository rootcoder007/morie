"""vmwav is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmwav import vmwav


def test_vmwav_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmwav()
