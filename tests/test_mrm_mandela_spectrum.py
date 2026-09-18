import pytest


def test_flagship_accepts_pandas_frames_csv_paths_and_rows(tmp_path):
    """The Mandela spectrum crashed on a real pandas frame (KeyError on a
    native boolean mask) and on a CSV path; every entry point now coerces."""
    import os

    import morie
    from morie.fn import _frame_core as fc
    from morie.mrm_mandela_spectrum import mrm_otis_mandela_spectrum as f
    p = os.path.join(os.path.dirname(morie.__file__), "data/samples/otis_b01_sample.csv")
    native = f(p)
    assert len(native) > 0
    rows = fc.read_csv(p)
    class FakePandas:  # duck-typed like pandas: .columns and column access with .tolist()
        def __init__(self, frame):
            self.columns = list(frame.columns); self._f = frame
        def __getitem__(self, c):
            class Col:
                def __init__(self, v): self.v = list(v)
                def tolist(self): return self.v
            return Col(self._f[c])
    fake = f(FakePandas(rows))
    assert fake.to_dict("records") == native.to_dict("records")
    pd = pytest.importorskip("pandas")
    real = f(pd.read_csv(p))
    assert real.to_dict("records") == native.to_dict("records")
    with pytest.raises(FileNotFoundError):
        f(str(tmp_path / "missing.csv"))
    with pytest.raises(TypeError):
        f(12345)
