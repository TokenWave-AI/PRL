"""The launchers' probe waits out a shared key's 429 and still fails on anything else."""

import io
import json
import urllib.error

import pytest

from prl.algo import llm_client


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "error", {}, io.BytesIO(b""))


def setup(monkeypatch, tmp_path, answers, wait=600):
    key = tmp_path / "key"
    key.write_text("k")
    for name, value in (("API_BASE", "http://x/v1"), ("KEY_FILE", str(key)), ("MODEL", "m")):
        monkeypatch.setenv(f"PRL_JUDGE_{name}", value)
    monkeypatch.setattr(llm_client, "PROBE_RATE_LIMIT_WAIT", wait)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)
    calls = iter(answers)

    def urlopen(request, timeout):
        answer = next(calls)
        if isinstance(answer, Exception):
            raise answer
        return io.BytesIO(json.dumps(answer).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)


MODELS, CHOICE = {"data": [{"id": "m"}]}, {"choices": [{}]}


def test_probe_waits_out_429(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, [http_error(429), MODELS, http_error(429), http_error(429), CHOICE])
    llm_client.probe()


def test_probe_fails_on_other_errors(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, [MODELS, http_error(403)])
    with pytest.raises(urllib.error.HTTPError):
        llm_client.probe()


def test_probe_gives_up_on_429_after_the_wait(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, [http_error(429)], wait=-1)
    with pytest.raises(urllib.error.HTTPError):
        llm_client.probe()


def test_probe_asks_every_model_of_the_chain(monkeypatch, tmp_path):
    monkeypatch.setenv("PRL_JUDGE_FALLBACK_MODELS", "f1,f2")
    setup(monkeypatch, tmp_path, [{"data": [{"id": "m"}, {"id": "f1"}, {"id": "f2"}]}, CHOICE, CHOICE, CHOICE])
    llm_client.probe()
    setup(monkeypatch, tmp_path, [{"data": [{"id": "m"}, {"id": "f1"}]}, CHOICE, CHOICE])
    with pytest.raises(AssertionError, match="f2 not served"):
        llm_client.probe()
