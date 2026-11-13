"""The shared settings file and the data service: no sign-in, no key."""

import json
import os
import stat

import pytest

from morie import datahub, hosted


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MORIE_HOSTED_KEY", raising=False)
    yield tmp_path


def test_credentials_are_written_owner_only(isolated_home):
    path = hosted.write_credentials({"api_base_url": "https://api.example.org/v1", "api_key": "k"})
    assert path == isolated_home / "morie" / "credentials.json"
    if os.name != "nt":  # NTFS has no POSIX mode bits; the file is still user-private there
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == stat.S_IRUSR | stat.S_IWUSR
    assert hosted.read_credentials()["api_key"] == "k"


def test_no_data_key_is_read_or_sent(isolated_home, monkeypatch):
    # a key an earlier version stored, and one in the environment: neither is read
    hosted.write_credentials({"hosted_key": "sk-stored", "hosted_user": "someone"})
    monkeypatch.setenv("MORIE_HOSTED_KEY", "sk-env")
    h = datahub._headers()
    assert list(h) == ["User-Agent"] and "sk-" not in json.dumps(h)
    for gone in ("hosted_key", "device_login", "email_login", "store_token", "logout", "signin_profile", "status"):
        assert not hasattr(hosted, gone), gone
    # the stored file is left as it was
    assert hosted.read_credentials()["hosted_key"] == "sk-stored"


def test_login_and_logout_are_not_commands():
    from morie.runner import build_parser

    for verb in ("login", "logout"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([verb])
