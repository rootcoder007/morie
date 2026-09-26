"""thmksp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.thmksp import thomas_cluster


def test_thmksp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        thomas_cluster(lambda_p=None, mu=None, sigma=None)
