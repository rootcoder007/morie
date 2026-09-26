"""sgtfid is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgtfid import sgt_fiedler_value


def test_sgtfid_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgt_fiedler_value(A=None)
