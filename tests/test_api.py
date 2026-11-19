import pytest
from fastapi.testclient import TestClient

from mindbridge.api import create_app
from mindbridge.config import Settings

PASSWORD = "a-long-enough-password"


def make_client(tmp_path, pipeline, **overrides):
    cfg = Settings(
        _env_file=None,
        embedding_backend="lsa",
        jwt_secret="k" * 40,
        db_path=tmp_path / "users.db",
        **overrides,
    )
    return TestClient(create_app(cfg, pipeline))


@pytest.fixture
def client(tmp_path, pipeline):
    with make_client(tmp_path, pipeline) as c:
        yield c


def login(client, name="hitesh"):
    client.post("/api/auth/register", json={"username": name, "password": PASSWORD})
    token = client.post("/api/auth/login", json={"username": name, "password": PASSWORD}).json()
    return {"Authorization": f"Bearer {token['access_token']}"}


def test_health_reports_configuration(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["auth_required"] is True
    assert body["chunks"] > 100


def test_chat_requires_a_token_when_auth_is_on(client):
    assert client.post("/api/chat", json={"message": "hello"}).status_code == 401
    bad = {"Authorization": "Bearer nonsense"}
    assert client.post("/api/chat", json={"message": "hello"}, headers=bad).status_code == 401


def test_full_conversation_flow(client, fake_llm):
    headers = login(client)
    first = client.post(
        "/api/chat", json={"message": "my mind races at night and I can't sleep"}, headers=headers
    ).json()
    assert first["mode"] == "grounded"
    assert first["sources"] and first["safety"]["level"] == "none"
    sid = first["session_id"]

    client.post(
        "/api/chat", json={"message": "what about caffeine?", "session_id": sid}, headers=headers
    )
    sent = fake_llm.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]

    assert client.delete(f"/api/sessions/{sid}", headers=headers).status_code == 204
    client.post(
        "/api/chat",
        json={"message": "how do I deal with exam stress", "session_id": sid},
        headers=headers,
    )
    assert len(fake_llm.calls[-1]["messages"]) == 1


def test_crisis_response_is_flagged_in_the_payload(client, fake_llm):
    headers = login(client)
    body = client.post(
        "/api/chat", json={"message": "I want to kill myself"}, headers=headers
    ).json()
    assert body["mode"] == "crisis"
    assert body["safety"]["level"] == "high"
    assert "14416" in body["reply"]
    assert fake_llm.calls == []


def test_users_cannot_read_each_others_sessions(client, fake_llm):
    alice, mallory = login(client, "alice"), login(client, "mallory")
    sid = client.post("/api/chat", json={"message": "I can't sleep"}, headers=alice).json()[
        "session_id"
    ]
    resp = client.post(
        "/api/chat", json={"message": "what did I say?", "session_id": sid}, headers=mallory
    )
    assert resp.status_code == 403
    assert client.delete(f"/api/sessions/{sid}", headers=mallory).status_code == 204
    again = client.post(
        "/api/chat", json={"message": "still here?", "session_id": sid}, headers=alice
    )
    assert again.status_code == 200


def test_message_validation(client):
    headers = login(client)
    assert client.post("/api/chat", json={"message": ""}, headers=headers).status_code == 422
    assert client.post("/api/chat", json={"message": "   "}, headers=headers).status_code == 422
    too_long = "x" * 2001
    assert client.post("/api/chat", json={"message": too_long}, headers=headers).status_code == 422


def test_registration_rules(client):
    weak = client.post("/api/auth/register", json={"username": "someone", "password": "short"})
    assert weak.status_code == 400
    client.post("/api/auth/register", json={"username": "someone", "password": PASSWORD})
    dupe = client.post("/api/auth/register", json={"username": "someone", "password": PASSWORD})
    assert dupe.status_code == 400
    wrong = client.post(
        "/api/auth/login", json={"username": "someone", "password": "nope-nope-nope"}
    )
    assert wrong.status_code == 401


def test_login_attempts_are_rate_limited(client):
    codes = [
        client.post(
            "/api/auth/login", json={"username": "ghost", "password": "whatever-1234"}
        ).status_code
        for _ in range(12)
    ]
    assert codes[:10] == [401] * 10
    assert 429 in codes[10:]


def test_chat_is_rate_limited_per_user(tmp_path, pipeline):
    with make_client(tmp_path, pipeline, rate_limit_per_min=3) as c:
        headers = login(c)
        codes = [
            c.post("/api/chat", json={"message": "hi"}, headers=headers).status_code
            for _ in range(5)
        ]
        assert codes == [200, 200, 200, 429, 429]


def test_open_mode_needs_no_token_and_has_no_auth_routes(tmp_path, pipeline):
    with make_client(tmp_path, pipeline, require_auth=False) as c:
        assert c.post("/api/chat", json={"message": "hello"}).status_code == 200
        assert (
            c.post("/api/auth/login", json={"username": "abc", "password": "x"}).status_code == 404
        )


def test_security_headers_and_ui_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "MindBridge" in page.text
    assert page.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200


def test_helplines_endpoint(client):
    body = client.get("/api/helplines").json()
    assert body["emergency"] == "112"
    assert any("14416" in line["numbers"] for line in body["lines"])


def test_prod_mode_refuses_a_weak_jwt_secret():
    with pytest.raises(ValueError):
        Settings(_env_file=None, env="prod", jwt_secret="change-me")
    assert Settings(_env_file=None, env="prod", jwt_secret="z" * 40).env == "prod"
