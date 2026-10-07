"""CKAN CSVs in Windows-1252 decode instead of failing (round-8 finding)."""

from morie.ingest import ckan


class _Resp:
    def __init__(self, content):
        self.content = content
        self.status_code = 200


class _Client:
    def __init__(self, content):
        self._content = content

    def get(self, url):
        return _Resp(self._content)


def _read(content, fmt="csv"):
    c = ckan.Client.__new__(ckan.Client)
    c._client = _Client(content)
    return c.read_resource("https://example.org/x." + fmt, as_format=fmt)


def test_windows_1252_csv_decodes():
    raw = "library,note\nBrantford,Children\x92s room\n".encode("latin-1")
    df = _read(raw)
    assert list(df["note"]) == ["Children’s room"]


def test_utf8_csv_unchanged():
    raw = "library,note\nSudbury,Bibliothèque\n".encode()
    assert list(_read(raw)["note"]) == ["Bibliothèque"]


def test_encoding_rule():
    assert ckan._text_encoding(b"plain ascii") == "utf-8"
    assert ckan._text_encoding("é".encode()) == "utf-8"
    assert ckan._text_encoding(b"caf\xe9") == "cp1252"
