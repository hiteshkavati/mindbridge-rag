from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    role: str  # "user" or "assistant"
    content: str


class _Session:
    __slots__ = ("owner", "turns", "touched")

    def __init__(self, owner: str, max_turns: int):
        self.owner = owner
        self.turns: deque[Turn] = deque(maxlen=max_turns)
        self.touched = time.monotonic()


class SessionStore:
    """In-memory conversation history with a TTL and a size cap.

    Nothing is written to disk. A session can only be read by the user that created it,
    so a guessed id is useless to anyone else.
    """

    def __init__(self, ttl_s: int = 7200, max_turns: int = 12, max_sessions: int = 5000):
        self._ttl = ttl_s
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._items: OrderedDict[str, _Session] = OrderedDict()
        self._lock = threading.Lock()

    def _evict(self) -> None:
        now = time.monotonic()
        for sid in [s for s, v in self._items.items() if now - v.touched > self._ttl]:
            del self._items[sid]
        while len(self._items) > self._max_sessions:
            self._items.popitem(last=False)

    def new_id(self) -> str:
        return secrets.token_urlsafe(16)

    def history(self, sid: str, owner: str) -> list[Turn]:
        with self._lock:
            self._evict()
            s = self._items.get(sid)
            return list(s.turns) if s and s.owner == owner else []

    def add(self, sid: str, owner: str, user: str, assistant: str) -> None:
        with self._lock:
            self._evict()
            s = self._items.get(sid)
            if s is None:
                s = self._items[sid] = _Session(owner, self._max_turns)
            elif s.owner != owner:
                raise PermissionError("session belongs to another user")
            s.turns.extend((Turn("user", user), Turn("assistant", assistant)))
            s.touched = time.monotonic()
            self._items.move_to_end(sid)

    def delete(self, sid: str, owner: str) -> bool:
        with self._lock:
            s = self._items.get(sid)
            if s is None or s.owner != owner:
                return False
            del self._items[sid]
            return True
