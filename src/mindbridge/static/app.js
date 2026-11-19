(() => {
  const $ = (id) => document.getElementById(id);
  const state = { token: null, sessionId: null, registering: false, authRequired: true };

  async function api(path, options = {}) {
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    if (state.token) headers.Authorization = `Bearer ${state.token}`;
    const res = await fetch(path, { ...options, headers });
    if (res.status === 401 && state.authRequired) {
      state.token = null;
      show("auth");
    }
    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
      throw new Error(typeof detail === "string" ? detail : detail[0]?.msg || "Invalid request");
    }
    return res.status === 204 ? null : res.json();
  }

  function show(which) {
    $("auth").hidden = which !== "auth";
    $("chat").hidden = which !== "chat";
    $("reset").hidden = which !== "chat";
    if (which === "chat") $("input").focus();
  }

  function addMessage(role, text, extra = {}) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}${extra.care ? " care" : ""}${extra.pending ? " pending" : ""}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);

    if (extra.sources && extra.sources.length) {
      const details = document.createElement("details");
      details.className = "sources";
      const summary = document.createElement("summary");
      summary.textContent = `Based on ${extra.sources.length} passage${extra.sources.length > 1 ? "s" : ""}`;
      const list = document.createElement("ol");
      for (const s of extra.sources) {
        const li = document.createElement("li");
        li.textContent = `${s.title}: ${s.snippet}`;
        list.appendChild(li);
      }
      details.append(summary, list);
      wrap.appendChild(details);
    }
    $("log").appendChild(wrap);
    $("log").scrollTop = $("log").scrollHeight;
    return wrap;
  }

  async function send(text) {
    addMessage("user", text);
    const pending = addMessage("bot", "Thinking...", { pending: true });
    $("send").disabled = true;
    try {
      const data = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ message: text, session_id: state.sessionId }),
      });
      state.sessionId = data.session_id;
      pending.remove();
      addMessage("bot", data.reply, {
        sources: data.sources,
        care: data.safety.level !== "none",
      });
    } catch (err) {
      pending.remove();
      addMessage("bot", `Sorry, something went wrong: ${err.message}`);
    } finally {
      $("send").disabled = false;
      $("input").focus();
    }
  }

  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("input").value.trim();
    if (!text || $("send").disabled) return;
    $("input").value = "";
    send(text);
  });

  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  });

  function setAuthMode(registering) {
    state.registering = registering;
    $("auth-title").textContent = registering ? "Create an account" : "Sign in";
    $("auth-submit").textContent = registering ? "Create account" : "Sign in";
    $("auth-toggle").textContent = registering ? "I already have an account" : "Create an account";
    $("password").autocomplete = registering ? "new-password" : "current-password";
    $("auth-error").textContent = "";
  }
  $("auth-toggle").addEventListener("click", () => setAuthMode(!state.registering));

  $("auth-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = JSON.stringify({ username: $("username").value, password: $("password").value });
    try {
      if (state.registering) await api("/api/auth/register", { method: "POST", body });
      const data = await api("/api/auth/login", { method: "POST", body });
      state.token = data.access_token;
      $("password").value = "";
      show("chat");
      greet();
    } catch (err) {
      $("auth-error").textContent = err.message;
    }
  });

  $("reset").addEventListener("click", async () => {
    if (state.sessionId) {
      try { await api(`/api/sessions/${state.sessionId}`, { method: "DELETE" }); } catch (_) { /* ignore */ }
    }
    state.sessionId = null;
    $("log").replaceChildren();
    greet();
  });

  function greet() {
    addMessage("bot", "Hi, I'm MindBridge. Tell me what's on your mind and I'll share what might help. " +
      "I'm an information tool, not a replacement for a counsellor or doctor.");
  }

  async function init() {
    try {
      const health = await api("/api/health");
      state.authRequired = health.auth_required;
      const h = await api("/api/helplines");
      const lines = h.lines.map((l) => `${l.name} ${l.numbers[0]}`).join(" · ");
      $("helplines").textContent = `In an emergency call ${h.emergency}. Support lines: ${lines}.`;
      if (state.authRequired) {
        setAuthMode(false);
        show("auth");
      } else {
        show("chat");
        greet();
      }
    } catch (_) {
      $("helplines").textContent = "Cannot reach the server. In an emergency call 112.";
    }
  }
  init();
})();
