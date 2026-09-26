"""glmmgmrf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmgmrf import glmmgmrf


def test_glmmgmrf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmgmrf()
