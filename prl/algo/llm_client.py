"""OpenAI-compatible chat calls for trajectory transforms (the ATLAS judge, later a rewriter, users' own functions).

``ask`` sends one chat request and parses the answer with a caller's function; HTTP errors, timeouts and answers the
parser rejects are retried, and HTTP 429 (a key's parallel-request cap), a lost connection and HTTP 5xx (the service
down or overloaded) wait without using an attempt, for up to RATE_LIMIT_WAIT per request. Requests in flight
are capped per event loop and profile: the rollout's trials share one cap. The key is read from a file and never
logged or recorded. ``probe`` is the launchers' check that the key, base URL and model work from this node.

Configuration: a profile is an env prefix, ``PRL_JUDGE_`` by default (scripts/run.sh forwards PRL_JUDGE_*):
    <prefix>API_BASE      OpenAI-compatible base URL (…/v1)
    <prefix>KEY_FILE      file holding the API key (never the key itself in the environment)
    <prefix>MODEL         model name, e.g. gpt-6.1-sol
    <prefix>FALLBACK_MODELS  comma-separated models asked in turn when the one before refuses the prompt (the
                          provider's prompt-policy filter; e.g. gpt-6-astra,gpt-5.6-sol after gpt-6.1-sol), each
                          with fresh attempts; default none. The record names the model that answered (``model``)
                          and those that refused (``refused_by``)
    <prefix>EXTRA_BODY    JSON object merged into every request (default ``{}``), e.g. ``{"reasoning_effort": "high"}``;
                          without it the provider's default decides
    <prefix>CONCURRENCY   requests in flight per event loop (default 64)
    <prefix>TIMEOUT       seconds per request (default 600); <prefix>ATTEMPTS (default 3)
    <prefix>MAX_TOKENS    ``max_tokens`` of every request (default 16384)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

JUDGE = "PRL_JUDGE_"
# HTTP 429, a lost connection and HTTP 5xx are retried for up to this long before they count as failed attempts. Without
# the wait for an unavailable service, a judge outage failed every trial's three attempts within a minute, and the
# transform's failure guard stopped the run after 32 trials in a row.
RATE_LIMIT_WAIT = 1800
# The launchers' probe waits this long on HTTP 429 (a job died on one, on a key shared with other users)
PROBE_RATE_LIMIT_WAIT = 600
# The judge's API gateway relays OpenAI's prompt-policy rejection as a 502 with this text. Asking again within the
# hour does not help (0 of 9 retries), although the verdict on a prompt can change from one day to the next
REFUSED = "flagged as potentially violating"

_slots: dict[tuple[asyncio.AbstractEventLoop, str], asyncio.Semaphore] = {}


def config(prefix: str = JUDGE) -> dict[str, Any]:
    extra_body = json.loads(os.getenv(f"{prefix}EXTRA_BODY") or "{}")
    if not isinstance(extra_body, dict):
        raise ValueError(f"{prefix}EXTRA_BODY must be a JSON object")  # noqa: TRY004 -- a config value
    fallbacks = [m.strip() for m in os.getenv(f"{prefix}FALLBACK_MODELS", "").split(",") if m.strip()]
    missing = [f"{prefix}{k}" for k in ("API_BASE", "MODEL", "KEY_FILE") if not os.getenv(f"{prefix}{k}")]
    if missing:
        raise ValueError(f"no model to call: set {', '.join(missing)} (the site profile's judge lines, "
                         "configs/sites/example.env)")
    key_file = Path(os.environ[f"{prefix}KEY_FILE"])
    if not key_file.is_file():
        raise ValueError(f"{prefix}KEY_FILE: no file {key_file}")
    return {"base": os.environ[f"{prefix}API_BASE"].rstrip("/"), "model": os.environ[f"{prefix}MODEL"],
            "fallback_models": fallbacks,
            "key": key_file.read_text().strip(),
            "concurrency": int(os.getenv(f"{prefix}CONCURRENCY", "64")),
            "timeout": float(os.getenv(f"{prefix}TIMEOUT", "600")),
            "attempts": int(os.getenv(f"{prefix}ATTEMPTS", "3")),
            "max_tokens": int(os.getenv(f"{prefix}MAX_TOKENS", "16384")), "extra_body": extra_body}


def probe(prefix: str = JUDGE) -> None:
    """Launcher check (no key in the output): the key file is readable, the API lists the model and its fallbacks,
    and each answers one short request with the configured extra body (Bailian lists models a key may not call: 403)."""
    import urllib.error
    import urllib.request

    def fetch(request: urllib.request.Request, timeout: int) -> dict[str, Any]:
        deadline = time.time() + PROBE_RATE_LIMIT_WAIT
        while True:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:  # a shared key at its parallel-request cap is no broken setup
                if exc.code != 429 or time.time() >= deadline:
                    raise
            time.sleep(random.uniform(5, 20))

    cfg = config(prefix)
    assert cfg["key"], f"empty {prefix}KEY_FILE"
    headers = {"Authorization": f"Bearer {cfg['key']}"}
    request = urllib.request.Request(f"{cfg['base']}/models", headers=headers)
    models = [m["id"] for m in fetch(request, 60)["data"]]
    for model in (cfg["model"], *cfg["fallback_models"]):
        assert model in models, f"{model} not served; the API lists {models}"
        body = {"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 1024,
                **cfg["extra_body"]}
        request = urllib.request.Request(f"{cfg['base']}/chat/completions", data=json.dumps(body).encode(),
                                         headers={**headers, "Content-Type": "application/json"})
        assert fetch(request, 300)["choices"], f"{model} returned no choice"  # HTTPError (e.g. 403) propagates


async def _post(cfg: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    import httpx

    body = {"model": cfg["model"], "messages": messages, "max_tokens": cfg["max_tokens"], **cfg["extra_body"]}
    async with httpx.AsyncClient(timeout=cfg["timeout"]) as client:
        response = await client.post(f"{cfg['base']}/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {cfg['key']}"})
    response.raise_for_status()
    return response.json()


def _unavailable(exc: Exception, status: int | None, body: str) -> bool:
    """A lost connection or a 5xx that is not a prompt refusal (REFUSED arrives as a 502 and goes to the next model)."""
    import httpx

    return isinstance(exc, httpx.TransportError) or (status is not None and status >= 500 and REFUSED not in body)


async def ask(messages: list[dict[str, str]], parse: Callable[[str], T], prefix: str = JUDGE,
              record: dict[str, Any] | None = None, feedback: bool = False) -> T | None:
    """``parse(content)`` of the first answer it accepts (it raises to reject one), or None after the attempts. With
    ``feedback``, a rejected answer and the reason go back to the model with the next attempt (asked the same question
    again, a model tends to repeat its mistake; in one test 2/3 rewrite plans were lost that way). The request's
    outcome goes into ``record``: latency_s, usage, answer, served_model, finish_reason of the last answer, errors,
    rejected (each rejected answer with its attempt and the reason the model was told),
    rate_limited, unavailable (waits for the service), refused (prompt-policy refusals) and usage_total over all answers; model (the one asked last: the
    one that answered, or the last of the chain) and refused_by (the models that refused, in order)."""
    cfg = config(prefix)
    record = {} if record is None else record
    errors, rate_limited, unavailable, refused, refused_by = [], 0, 0, 0, []
    rejected: list[dict[str, Any]] = []  # answers that came and were rejected, with the reason
    chain = [cfg["model"], *cfg["fallback_models"]]
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    conversation = list(messages)
    slots = _slots.setdefault((asyncio.get_running_loop(), prefix), asyncio.Semaphore(cfg["concurrency"]))
    result = None
    async with slots:
        attempt, deadline = 0, time.time() + RATE_LIMIT_WAIT
        while attempt < cfg["attempts"]:
            t0 = time.time()
            content = None
            try:
                answer = await _post({**cfg, "model": chain[len(refused_by)]}, conversation)
                content = answer["choices"][0]["message"].get("content") or ""
                record.update(latency_s=round(time.time() - t0, 1), usage=answer.get("usage"), answer=content,
                              served_model=answer.get("model"), finish_reason=answer["choices"][0].get("finish_reason"))
                for key in total:
                    total[key] += (answer.get("usage") or {}).get(key) or 0
                result = parse(content)
            except Exception as exc:  # noqa: BLE001 -- HTTP, timeout, rejected answer: retry
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 429 and time.time() < deadline:
                    rate_limited += 1  # the key's parallel-request cap: wait, without using an attempt
                    await asyncio.sleep(random.uniform(5, 20))
                    continue
                body = getattr(getattr(exc, "response", None), "text", "") or ""
                if _unavailable(exc, status, body) and time.time() < deadline:
                    unavailable += 1  # the service is down or overloaded: wait for it, without using an attempt
                    await asyncio.sleep(random.uniform(10, 30))
                    continue
                errors.append(f"{chain[len(refused_by)]} attempt {attempt}: {exc!r} {body[:300]}"[:800])
                if REFUSED in body:
                    refused += 1
                    refused_by.append(chain[len(refused_by)])
                    if len(refused_by) < len(chain):  # the next model of the chain, from a fresh start
                        attempt, conversation = 0, list(messages)
                        continue
                    break
                if content is not None:
                    rejected.append({"attempt": attempt, "answer": content, "reason": str(exc)})
                if feedback and content is not None:  # an answer came and was rejected: say why
                    conversation = [*messages, {"role": "assistant", "content": content},
                                    {"role": "user", "content": f"That answer was rejected: {exc}\nAnswer again, in "
                                                                "the same format, with this fixed."}]
                await asyncio.sleep(min(30, 5 * 2**attempt))
                attempt += 1
                continue
            break
    record.update(errors=errors, rejected=rejected, rate_limited=rate_limited, unavailable=unavailable, refused=refused,
                  usage_total=total,
                  model=chain[min(len(refused_by), len(chain) - 1)], refused_by=refused_by)
    return result
