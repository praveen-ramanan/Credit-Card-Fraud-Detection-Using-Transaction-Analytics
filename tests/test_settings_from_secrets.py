"""BDA_* Streamlit secrets reach os.environ before config is imported
(src/dashboard/settings_from_secrets.py). Each case runs in a fresh interpreter,
because Streamlit's secrets object and config are process-wide."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
ROOT = Path(__file__).resolve().parent.parent

SCRIPT = f"""
import os, sys
sys.path.insert(0, {str(ROOT)!r})
os.environ["BDA_MONGO_DB"] = "from_env"
from src.dashboard.settings_from_secrets import promote_secrets_to_env
print(sorted(promote_secrets_to_env()))
from config import config as cfg
print(cfg.MONGO_URI_CONFIGURED, cfg.MONGO_URI, cfg.MONGO_DB_NAME, cfg.MONGO_ALLOW_MOCK_FALLBACK)
"""


def _run(tmp_path, secrets_toml=None):
    if secrets_toml is not None:
        (tmp_path / ".streamlit").mkdir()
        (tmp_path / ".streamlit" / "secrets.toml").write_text(secrets_toml)
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("BDA_")}
    env["HOME"] = env["USERPROFILE"] = str(tmp_path)  # ignore the user's ~/.streamlit/secrets.toml
    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-2:]


def test_secrets_become_settings(tmp_path):
    promoted, settings = _run(tmp_path, 'BDA_MONGO_URI = "mongodb://db.example:27017"\n'
                                        'BDA_MONGO_FALLBACK = false\nBDA_MONGO_DB = "from_secrets"\n'
                                        'OTHER = "ignored"\n[section]\nBDA_X = "ignored"\n')
    # string secrets are copied by Streamlit itself (overriding the environment);
    # the helper adds the boolean, which Streamlit skips
    assert promoted == "['BDA_MONGO_FALLBACK']"
    assert settings == "True mongodb://db.example:27017 from_secrets False"


def test_no_secrets_file_is_fine(tmp_path):
    promoted, settings = _run(tmp_path)
    assert promoted == "[]"
    assert settings == "False mongodb://localhost:27017 from_env True"
