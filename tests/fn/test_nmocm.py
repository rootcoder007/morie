"""nmocm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmocm import oc_coombs_mesh


def test_nmocm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        oc_coombs_mesh(data=None)
