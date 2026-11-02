"""
Streamlit secrets -> environment variables, for the BDA_* settings that
config/config.py reads from os.environ.

Streamlit (including Community Cloud) copies root-level string/number
secrets into os.environ whenever it parses secrets.toml - overwriting an
environment variable of the same name. This adds what it skips: TOML
booleans (e.g. `BDA_MONGO_FALLBACK = false`). It also makes sure the file
is parsed before `config` is imported, whatever launched the app.
"""

import os

import streamlit as st

PREFIX = "BDA_"


def promote_secrets_to_env() -> list:
    """Sets every root-level BDA_* secret that is not an environment variable
    yet (after Streamlit's own copy). Returns the names it set. No secrets
    file is not an error."""
    try:
        if not st.secrets.load_if_toml_exists():
            return []
        items = st.secrets.to_dict()
    except Exception as e:  # malformed secrets.toml: the app still starts on its defaults
        print(f"[settings_from_secrets] could not read Streamlit secrets: {e}")
        return []

    promoted = []
    for key, value in items.items():
        if not key.startswith(PREFIX) or key in os.environ:
            continue
        if isinstance(value, bool):
            value = "1" if value else "0"
        elif not isinstance(value, (str, int, float)):
            continue  # tables/lists are not settings
        os.environ[key] = str(value)
        promoted.append(key)
    return promoted
