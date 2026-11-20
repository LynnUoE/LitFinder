from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest

import api_server
from health import LLMKeyCheck


def api_error(cls, status: int, message: str):
    request = httpx.Request("GET", "https://fake.test/v1/models")
    return cls(message, response=httpx.Response(status, request=request), body=None)


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def list(self):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_check(*outcomes, interval=300):
    models = FakeModels(outcomes)
    clock = Clock()
    return LLMKeyCheck(lambda: SimpleNamespace(models=models), interval=interval, clock=clock), models, clock


def test_valid_key_is_ok_and_cached_for_the_interval():
    check, models, clock = make_check([], [])
    assert check.status()["status"] == "ok"
    clock.now += 299
    assert check.status()["status"] == "ok" and models.calls == 1
    clock.now += 1
    check.status()
    assert models.calls == 2


@pytest.mark.parametrize("cls, status", [(openai.AuthenticationError, 401), (openai.PermissionDeniedError, 403)])
def test_rejected_key_is_invalid_without_quoting_the_provider(cls, status):
    secret = "Incorrect API key provided: sk-proj-****abcd"
    check, _, _ = make_check(api_error(cls, status, secret))
    result = check.status()
    assert result["status"] == "invalid_key"
    assert f"HTTP {status}" in result["message"] and "sk-" not in result["message"]


@pytest.mark.parametrize("error", [
    openai.APITimeoutError(request=httpx.Request("GET", "https://fake.test/v1/models")),
    api_error(openai.NotFoundError, 404, "no models endpoint"),
    api_error(openai.InternalServerError, 500, "outage"),
])
def test_other_failures_are_inconclusive(error):
    check, _, _ = make_check(error)
    result = check.status()
    assert result["status"] == "unknown" and type(error).__name__ in result["message"]


def test_client_is_created_once_and_a_failing_factory_is_inconclusive():
    created = []

    def factory():
        created.append(1)
        if len(created) == 1:
            raise ValueError("bad base URL")
        return SimpleNamespace(models=FakeModels([[], []]))

    clock = Clock()
    check = LLMKeyCheck(factory, interval=1, clock=clock)
    assert check.status()["status"] == "unknown"
    clock.now += 1
    assert check.status()["status"] == "ok"
    clock.now += 1
    assert check.status()["status"] == "ok" and len(created) == 2


@pytest.mark.parametrize("outcome, http_status, status", [
    ([], 200, "healthy"),
    (api_error(openai.AuthenticationError, 401, "expired"), 503, "unhealthy"),
    (api_error(openai.InternalServerError, 500, "outage"), 200, "healthy"),
])
def test_health_endpoint_reports_the_llm_check(monkeypatch, outcome, http_status, status):
    check, _, _ = make_check(outcome)
    monkeypatch.setattr(api_server, "llm_key_check", check)
    response = api_server.app.test_client().get("/api/health_check")
    assert response.status_code == http_status
    body = response.get_json()
    assert body["status"] == status and body["checks"]["llm"]["status"] in ("ok", "invalid_key", "unknown")


def test_health_endpoint_without_the_check(monkeypatch):
    monkeypatch.setattr(api_server, "llm_key_check", None)
    response = api_server.app.test_client().get("/api/health_check")
    assert response.status_code == 200 and response.get_json()["checks"]["llm"] == {"status": "not_checked"}
