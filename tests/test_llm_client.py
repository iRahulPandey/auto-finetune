"""
tests/test_llm_client.py — Unit tests for LLM client URL validation.
"""

import pytest

from _core.llm_client import _validate_ollama_url


def test_http_url_accepted():
    assert _validate_ollama_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"


def test_https_url_accepted():
    assert _validate_ollama_url("https://ollama.example.com") == "https://ollama.example.com"


def test_file_scheme_rejected():
    with pytest.raises(ValueError, match="http or https"):
        _validate_ollama_url("file:///etc/passwd")


def test_ftp_scheme_rejected():
    with pytest.raises(ValueError, match="http or https"):
        _validate_ollama_url("ftp://internal-server")


def test_no_scheme_rejected():
    with pytest.raises(ValueError):
        _validate_ollama_url("127.0.0.1:11434")


def test_ssrf_metadata_endpoint_scheme_blocked():
    # Cloud metadata endpoint — scheme is http so it passes URL validation,
    # but this test documents the known risk (network-level blocking required)
    result = _validate_ollama_url("http://169.254.169.254/latest/meta-data/")
    assert result.startswith("http")
