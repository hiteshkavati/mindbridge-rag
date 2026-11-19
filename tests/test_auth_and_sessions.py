import time

import jwt
import pytest

from mindbridge.auth import (
    AuthError,
    UserStore,
    hash_password,
    issue_token,
    read_token,
    verify_password,
)
from mindbridge.sessions import SessionStore


def test_password_hash_is_salted_and_verifiable():
    a, b = hash_password("correct horse battery"), hash_password("correct horse battery")
    assert a != b
    assert a.startswith("scrypt$")
    assert verify_password("correct horse battery", a)
    assert not verify_password("wrong password!!", a)


@pytest.mark.parametrize("stored", ["", "garbage", "md5$1$2$3$4$5", "scrypt$x$y$z$q$w"])
def test_malformed_stored_hash_never_verifies(stored):
    assert not verify_password("anything", stored)


def test_register_and_authenticate(tmp_path):
    store = UserStore(tmp_path / "u.db")
    assert store.register("  Hitesh ", "a-long-enough-password") == "hitesh"
    assert store.authenticate("HITESH", "a-long-enough-password") == "hitesh"
    with pytest.raises(AuthError):
        store.authenticate("hitesh", "not the password")
    with pytest.raises(AuthError):
        store.authenticate("nobody", "a-long-enough-password")


def test_duplicate_and_invalid_registrations_fail(tmp_path):
    store = UserStore(tmp_path / "u.db")
    store.register("alice", "a-long-enough-password")
    with pytest.raises(AuthError):
        store.register("Alice", "another-long-password")
    with pytest.raises(AuthError):
        store.register("x", "a-long-enough-password")
    with pytest.raises(AuthError):
        store.register("bob", "short")
    with pytest.raises(AuthError):
        store.register("bad name!", "a-long-enough-password")


def test_passwords_are_not_stored_in_plaintext(tmp_path):
    import sqlite3

    UserStore(tmp_path / "u.db").register("carol", "plain-text-check-123")
    raw = sqlite3.connect(tmp_path / "u.db").execute("SELECT pw_hash FROM users").fetchone()[0]
    assert "plain-text-check-123" not in raw


def test_token_round_trip_and_tampering():
    secret = "s" * 40
    token = issue_token("dave", secret, 5)
    assert read_token(token, secret) == "dave"
    with pytest.raises(AuthError):
        read_token(token, "t" * 40)
    with pytest.raises(AuthError):
        read_token(token + "x", secret)


def test_expired_token_is_rejected():
    secret = "s" * 40
    expired = jwt.encode({"sub": "eve", "exp": int(time.time()) - 5}, secret, algorithm="HS256")
    with pytest.raises(AuthError):
        read_token(expired, secret)


def test_token_without_subject_is_rejected():
    secret = "s" * 40
    token = jwt.encode({"exp": int(time.time()) + 60}, secret, algorithm="HS256")
    with pytest.raises(AuthError):
        read_token(token, secret)


def test_session_history_is_private_to_its_owner():
    store = SessionStore()
    sid = store.new_id()
    store.add(sid, "alice", "hi", "hello")
    assert [t.content for t in store.history(sid, "alice")] == ["hi", "hello"]
    assert store.history(sid, "mallory") == []
    with pytest.raises(PermissionError):
        store.add(sid, "mallory", "x", "y")
    assert not store.delete(sid, "mallory")
    assert store.delete(sid, "alice")
    assert store.history(sid, "alice") == []


def test_session_keeps_only_recent_turns():
    store = SessionStore(max_turns=4)
    for i in range(5):
        store.add("s", "u", f"q{i}", f"a{i}")
    assert [t.content for t in store.history("s", "u")] == ["q3", "a3", "q4", "a4"]


def test_sessions_expire(monkeypatch):
    store = SessionStore(ttl_s=10)
    store.add("s", "u", "q", "a")
    real = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: real + 60)
    assert store.history("s", "u") == []


def test_session_count_is_capped():
    store = SessionStore(max_sessions=3)
    for i in range(6):
        store.add(f"s{i}", "u", "q", "a")
    assert store.history("s0", "u") == []
    assert store.history("s5", "u")
