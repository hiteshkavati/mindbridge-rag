from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import jwt

_USERNAME = re.compile(r"^[a-z0-9_.-]{3,32}$")
_N, _R, _P = 2**14, 8, 1


class AuthError(Exception):
    pass


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(
            password.encode(),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# Verifying against this when the user does not exist keeps response time similar,
# so the login endpoint does not reveal which usernames are registered.
_DUMMY_HASH = hash_password("not-a-real-password")


def normalise_username(username: str) -> str:
    name = username.strip().lower()
    if not _USERNAME.match(name):
        raise AuthError("username must be 3-32 characters: letters, digits, dot, dash, underscore")
    return name


def check_password_strength(password: str) -> None:
    if not 10 <= len(password) <= 128:
        raise AuthError("password must be between 10 and 128 characters")


class UserStore:
    def __init__(self, path: Path | str):
        self._path = str(path)
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS users ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "username TEXT NOT NULL UNIQUE,"
                "pw_hash TEXT NOT NULL,"
                "created_at REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def register(self, username: str, password: str) -> str:
        name = normalise_username(username)
        check_password_strength(password)
        try:
            with closing(self._connect()) as db, db:
                db.execute(
                    "INSERT INTO users (username, pw_hash, created_at) VALUES (?, ?, ?)",
                    (name, hash_password(password), time.time()),
                )
        except sqlite3.IntegrityError as exc:
            raise AuthError("username is taken") from exc
        return name

    def authenticate(self, username: str, password: str) -> str:
        name = username.strip().lower()
        with closing(self._connect()) as db:
            row = db.execute("SELECT pw_hash FROM users WHERE username = ?", (name,)).fetchone()
        ok = verify_password(password, row[0] if row else _DUMMY_HASH)
        if not row or not ok:
            raise AuthError("invalid username or password")
        return name


def issue_token(username: str, secret: str, ttl_minutes: int) -> str:
    now = int(time.time())
    claims = {"sub": username, "iat": now, "exp": now + ttl_minutes * 60}
    return jwt.encode(claims, secret, algorithm="HS256")


def read_token(token: str, secret: str) -> str:
    try:
        claims = jwt.decode(
            token, secret, algorithms=["HS256"], options={"require": ["exp", "sub"]}
        )
    except jwt.PyJWTError as exc:
        raise AuthError("invalid or expired token") from exc
    return str(claims["sub"])
