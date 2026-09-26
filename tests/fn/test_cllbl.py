"""cllbl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cllbl import cllbl


def test_cllbl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cllbl()
