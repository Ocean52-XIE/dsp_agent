const state = {
  sessions: [],
  currentSession: null,
  selectedMessageId: null,
  inlineCitationSelection: {},
  inlinePanelExpanded: {},
  debugVerboseEnabled: false,
  composerHintTimer: null,
  sending: false,
  sidebarCollapsed: false,
};

const SESSION_LIST_LIMIT = 20;
const INLINE_DEBUG_VALUE_MAX_CHARS = 88;
const DEFAULT_COMPOSER_HINT_TEXT = "";
const PENDING_BLOCKED_HINT_TEXT = "上一个问题还在思考中，请稍后";
const ASSISTANT_THINKING_TEXT = "系统思考中。。。";

const elements = {
  appShell: document.getElementById("appShell"),
  contentGrid: document.getElementById("contentGrid"),
  sidebar: document.querySelector(".sidebar"),
  sidebarToggleBtn: document.getElementById("sidebarToggleBtn"),
  sidebarCollapsedActions: document.getElementById("sidebarCollapsedActions"),
  sidebarToggleMiniBtn: document.getElementById("sidebarToggleMiniBtn"),
  newSessionMiniBtn: document.getElementById("newSessionMiniBtn"),
  newSessionBtn: document.getElementById("newSessionBtn"),
  sessionCountTag: document.getElementById("sessionCountTag"),
  sessionList: document.getElementById("sessionList"),
  routeBadge: document.getElementById("routeBadge"),
  statusBadge: document.getElementById("statusBadge"),
  messageList: document.getElementById("messageList"),
  composer: document.querySelector(".composer"),
  composerForm: document.getElementById("composerForm"),
  composerInput: document.getElementById("composerInput"),
  composerHint: document.getElementById("composerHint"),
  sendButton: document.getElementById("sendButton"),
};

document.addEventListener("DOMContentLoaded", () => {
  restoreSidebarState();
  bindEvents();
  bootstrap().catch(handleError);
});

function bindEvents() {
  if (elements.sidebarToggleBtn) {
    elements.sidebarToggleBtn.addEventListener("click", () => {
      setSidebarCollapsed(!state.sidebarCollapsed);
    });
  }

  if (elements.sidebarToggleMiniBtn) {
    elements.sidebarToggleMiniBtn.addEventListener("click", () => {
      setSidebarCollapsed(!state.sidebarCollapsed);
    });
  }

  if (elements.newSessionBtn) {
    elements.newSessionBtn.addEventListener("click", () => {
      beginDraftSession();
    });
  }

  if (elements.newSessionMiniBtn) {
    elements.newSessionMiniBtn.addEventListener("click", () => {
      beginDraftSession();
      setSidebarCollapsed(false);
    });
  }

  if (elements.composerForm) {
    elements.composerForm.addEventListener("submit", (event) => {
      event.preventDefault();
      submitMessage().catch(handleError);
    });
  }

  if (elements.composerInput) {
    elements.composerInput.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") {
        return;
      }
      if (event.isComposing || event.keyCode === 229) {
        return;
      }
      if (event.ctrlKey) {
        event.preventDefault();
        insertComposerNewlineAtCursor();
        return;
      }
      event.preventDefault();
      submitMessage().catch(handleError);
    });
  }

  if (elements.sidebar && elements.sessionList) {
    elements.sidebar.addEventListener("wheel", onSidebarWheel, { passive: false });
  }

  if (elements.composer && elements.messageList) {
    elements.composer.addEventListener("wheel", onComposerWheel, { passive: false });
  }

  document.addEventListener("click", (event) => {
    const sessionButton = event.target.closest("[data-session-id]");
    if (sessionButton) {
      openSession(sessionButton.dataset.sessionId).catch(handleError);
      return;
    }

    const copyCitationCodeButton = event.target.closest("[data-action='copy-citation-code']");
    if (copyCitationCodeButton) {
      copyCitationCode(copyCitationCodeButton).catch(() => {
        // 复制失败时不打断主流程，只保持按钮原状态。
      });
      return;
    }

    const copyCitationTextButton = event.target.closest("[data-action='copy-citation-text']");
    if (copyCitationTextButton) {
      copyCitationText(copyCitationTextButton).catch(() => {
        // 复制失败时不打断主流程，只保持按钮原状态。
      });
      return;
    }

    const citationSwitchButton = event.target.closest("[data-action='select-inline-citation']");
    if (citationSwitchButton) {
      const messageId = citationSwitchButton.dataset.messageId || "";
      const citationIndex = Number.parseInt(citationSwitchButton.dataset.citationIndex || "0", 10);
      selectInlineCitation(messageId, citationIndex);
      return;
    }

    const inlinePanelToggleButton = event.target.closest("[data-action='toggle-inline-panel']");
    if (inlinePanelToggleButton) {
      const messageId = inlinePanelToggleButton.dataset.messageId || "";
      const panelType = inlinePanelToggleButton.dataset.panelType || "";
      toggleInlinePanel(messageId, panelType);
      return;
    }

  });
}

function onSidebarWheel(event) {
  if (window.matchMedia("(max-width: 1180px)").matches || state.sidebarCollapsed) {
    return;
  }
  const list = elements.sessionList;
  if (!list || Math.abs(event.deltaY) < 0.1) {
    return;
  }
  event.preventDefault();
  list.scrollTop += event.deltaY;
}

function onComposerWheel(event) {
  if (window.matchMedia("(max-width: 1180px)").matches) {
    return;
  }
  const list = elements.messageList;
  if (!list || Math.abs(event.deltaY) < 0.1) {
    return;
  }
  const overflowY = window.getComputedStyle(list).overflowY;
  const canScrollMessageList = overflowY !== "visible" && list.scrollHeight > list.clientHeight + 1;
  if (!canScrollMessageList) {
    return;
  }
  event.preventDefault();
  list.scrollTop += event.deltaY;
}

async function copyCitationCode(button) {
  const codeWrap = button?.closest(".citation-excerpt-code-wrap");
  const codeElement = codeWrap?.querySelector("code");
  const codeText = codeElement?.textContent || "";
  if (!codeText.trim()) {
    return;
  }

  const copied = await copyTextToClipboard(codeText);
  applyCopyButtonFeedback(button, copied);
}

async function copyCitationText(button) {
  const rawText = String(button?.dataset?.copyText || "");
  if (!rawText.trim()) {
    return;
  }
  const copied = await copyTextToClipboard(rawText);
  applyCopyButtonFeedback(button, copied);
}

async function copyTextToClipboard(rawText) {
  const text = String(rawText || "");
  if (!text.trim()) {
    return false;
  }

  if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_error) {
      // fallback
    }
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch (_error) {
    copied = false;
  }
  document.body.removeChild(textarea);
  return copied;
}

function applyCopyButtonFeedback(button, copied) {
  if (!button) {
    return;
  }
  const originalLabel = button.dataset.originLabel || button.textContent || "复制";
  button.dataset.originLabel = originalLabel;
  button.textContent = copied ? "已复制" : "复制失败";
  button.classList.toggle("copied", copied);
  button.classList.toggle("copy-failed", !copied);
  window.setTimeout(() => {
    button.textContent = originalLabel;
    button.classList.remove("copied");
    button.classList.remove("copy-failed");
  }, 1300);
}

function insertComposerNewlineAtCursor() {
  const input = elements.composerInput;
  if (!input) {
    return;
  }

  const start = Number.isInteger(input.selectionStart) ? input.selectionStart : input.value.length;
  const end = Number.isInteger(input.selectionEnd) ? input.selectionEnd : input.value.length;
  const before = input.value.slice(0, start);
  const after = input.value.slice(end);
  input.value = `${before}\n${after}`;

  const nextCursor = start + 1;
  input.selectionStart = nextCursor;
  input.selectionEnd = nextCursor;
}

function showComposerHint(text, options = {}) {
  const { warning = false, autoReset = false, ttlMs = 2200 } = options;
  if (!elements.composerHint) {
    return;
  }
  const hintText = text || DEFAULT_COMPOSER_HINT_TEXT;
  elements.composerHint.textContent = hintText;
  elements.composerHint.hidden = hintText.trim().length === 0;
  elements.composerHint.classList.toggle("warning", warning);

  if (state.composerHintTimer !== null) {
    clearTimeout(state.composerHintTimer);
    state.composerHintTimer = null;
  }
  if (!autoReset) {
    return;
  }
  state.composerHintTimer = window.setTimeout(() => {
    if (!elements.composerHint) {
      return;
    }
    elements.composerHint.textContent = DEFAULT_COMPOSER_HINT_TEXT;
    elements.composerHint.hidden = DEFAULT_COMPOSER_HINT_TEXT.trim().length === 0;
    elements.composerHint.classList.remove("warning");
    state.composerHintTimer = null;
  }, ttlMs);
}

function isLocalSessionId(sessionId) {
  return typeof sessionId === "string" && sessionId.startsWith("local-session-");
}

function appendOptimisticPendingTurn(content) {
  const nowIso = new Date().toISOString();
  if (!state.currentSession) {
    state.currentSession = {
      id: `local-session-${Date.now()}`,
      title: "新会话",
      status: "processing",
      updated_at: nowIso,
      messages: [],
    };
    state.inlineCitationSelection = {};
    state.inlinePanelExpanded = {};
  }
  if (!Array.isArray(state.currentSession.messages)) {
    state.currentSession.messages = [];
  }

  const userMessageId = `local-user-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  const assistantMessageId = `local-assistant-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  const userMessage = {
    id: userMessageId,
    role: "user",
    content,
    created_at: nowIso,
  };
  const assistantMessage = {
    id: assistantMessageId,
    role: "assistant",
    content: ASSISTANT_THINKING_TEXT,
    created_at: nowIso,
    intent: "knowledge_qa",
    status: "processing",
    analysis: {
      generation_mode: "pending",
    },
  };

  state.currentSession.messages.push(userMessage, assistantMessage);
  state.currentSession.status = "processing";
  state.currentSession.updated_at = nowIso;
  state.selectedMessageId = assistantMessageId;
  renderAll();
  scrollMessagesToBottom();
  return { userMessageId, assistantMessageId };
}

function markOptimisticAssistantAsFailed(assistantMessageId) {
  if (!assistantMessageId || !state.currentSession || !Array.isArray(state.currentSession.messages)) {
    return;
  }
  const target = state.currentSession.messages.find((item) => item.id === assistantMessageId);
  if (!target) {
    return;
  }
  target.content = "系统响应失败，请稍后重试。";
  target.status = "error";
  target.analysis = {
    ...(target.analysis || {}),
    generation_mode: "fallback",
    llm_fallback_reason: "request_failed",
  };
  state.currentSession.status = "error";
  state.selectedMessageId = assistantMessageId;
  renderAll();
}

function selectInlineCitation(messageId, citationIndex) {
  if (!messageId) {
    return;
  }
  if (!Number.isInteger(citationIndex) || citationIndex < 0) {
    return;
  }
  state.inlineCitationSelection[messageId] = citationIndex;
  renderMessages();
}

function getInlinePanelKey(messageId, panelType) {
  return `${messageId}:${panelType}`;
}

function isInlinePanelExpanded(messageId, panelType) {
  if (!messageId || !panelType) {
    return false;
  }
  return state.inlinePanelExpanded[getInlinePanelKey(messageId, panelType)] === true;
}

function toggleInlinePanel(messageId, panelType) {
  if (!messageId || !panelType) {
    return;
  }
  const key = getInlinePanelKey(messageId, panelType);
  state.inlinePanelExpanded[key] = !state.inlinePanelExpanded[key];
  renderMessages();
}

function restoreSidebarState() {
  try {
    const cached = window.localStorage.getItem("sidebar_collapsed");
    setSidebarCollapsed(cached === "1");
  } catch (_error) {
    setSidebarCollapsed(false);
  }
}

function setSidebarCollapsed(collapsed) {
  state.sidebarCollapsed = Boolean(collapsed);

  if (elements.appShell) {
    elements.appShell.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
  }
  if (elements.sidebarCollapsedActions) {
    elements.sidebarCollapsedActions.setAttribute("aria-hidden", state.sidebarCollapsed ? "false" : "true");
  }
  if (elements.sidebarToggleBtn) {
    elements.sidebarToggleBtn.setAttribute("aria-label", state.sidebarCollapsed ? "展开会话栏" : "隐藏会话栏");
    elements.sidebarToggleBtn.setAttribute("title", state.sidebarCollapsed ? "展开会话栏" : "隐藏会话栏");
  }
  if (elements.sidebarToggleMiniBtn) {
    elements.sidebarToggleMiniBtn.setAttribute("aria-label", state.sidebarCollapsed ? "展开会话栏" : "隐藏会话栏");
    elements.sidebarToggleMiniBtn.setAttribute("title", state.sidebarCollapsed ? "展开会话栏" : "隐藏会话栏");
  }

  try {
    window.localStorage.setItem("sidebar_collapsed", state.sidebarCollapsed ? "1" : "0");
  } catch (_error) {
    // ignore
  }
}

async function bootstrap() {
  if (elements.messageList) {
    elements.messageList.innerHTML = `<div class="loading-shell">正在加载会话数据...</div>`;
  }
  await refreshHealth();
  await refreshSessions();
  beginDraftSession();
}

async function refreshHealth() {
  try {
    const payload = await api("/api/health");
    state.debugVerboseEnabled = Boolean(payload?.debug_verbose_enabled);
  } catch (_error) {
    state.debugVerboseEnabled = false;
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let errorText = `请求失败: ${response.status}`;
    try {
      const payload = await response.json();
      errorText = payload.detail || errorText;
    } catch (_error) {
      // ignore
    }
    throw new Error(errorText);
  }
  return response.json();
}

async function refreshSessions() {
  const payload = await api(`/api/sessions?limit=${SESSION_LIST_LIMIT}`);
  state.sessions = payload.items || [];
  renderSessionList();
}

function beginDraftSession() {
  if (state.sending) {
    return;
  }
  state.currentSession = null;
  state.selectedMessageId = null;
  state.inlineCitationSelection = {};
  state.inlinePanelExpanded = {};
  if (elements.composerInput) {
    elements.composerInput.value = "";
    elements.composerInput.focus();
  }
  renderAll();
}

async function openSession(sessionId) {
  if (!sessionId) {
    return;
  }
  const payload = await api(`/api/sessions/${sessionId}`);
  state.currentSession = payload.session;
  state.inlineCitationSelection = {};
  state.inlinePanelExpanded = {};
  state.selectedMessageId = resolveSelectedMessageId(state.currentSession);
  renderAll();
  scrollMessagesToBottom();
}

async function submitMessage() {
  const rawContent = elements.composerInput?.value || "";
  const content = rawContent.trim();
  if (!content) {
    return;
  }
  if (state.sending) {
    showComposerHint(PENDING_BLOCKED_HINT_TEXT, { warning: true, autoReset: true });
    return;
  }

  if (elements.composerInput) {
    elements.composerInput.value = "";
  }

  const optimisticTurn = appendOptimisticPendingTurn(content);

  setPending(true);
  try {
    if (!state.currentSession || isLocalSessionId(state.currentSession.id)) {
      const created = await api("/api/sessions", {
        method: "POST",
        body: JSON.stringify({}),
      });
      if (!state.currentSession) {
        state.currentSession = created.session;
      } else {
        state.currentSession.id = created.session.id;
        state.currentSession.title = created.session.title || state.currentSession.title;
      }
    }

    const payload = await api("/api/messages", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.currentSession.id,
        content,
      }),
    });

    await refreshSessions();
    state.currentSession = payload.session;
    state.selectedMessageId = resolveSelectedMessageId(state.currentSession, payload.assistant_message_id);
    renderAll();
    scrollMessagesToBottom();
  } catch (error) {
    markOptimisticAssistantAsFailed(optimisticTurn.assistantMessageId);
    throw error;
  } finally {
    setPending(false);
  }
}

function renderAll() {
  applyDraftComposerModeClass();
  renderSessionList();
  renderWorkspace();
  renderMessages();
}

function isDraftComposerMode() {
  if (!state.currentSession) {
    return true;
  }
  const visibleMessages = getVisibleMessages(state.currentSession.messages);
  return visibleMessages.length === 0;
}

function applyDraftComposerModeClass() {
  if (!elements.appShell) {
    return;
  }
  elements.appShell.classList.toggle("draft-composer-mode", isDraftComposerMode());
}

function renderSessionList() {
  if (!elements.sessionList || !elements.sessionCountTag) {
    return;
  }

  elements.sessionCountTag.textContent = String(state.sessions.length);
  if (state.sessions.length === 0) {
    elements.sessionList.innerHTML = `<div class="empty-state">暂无会话，发送首条消息后会自动创建。</div>`;
    return;
  }

  const groups = groupSessionsByAge(state.sessions);
  elements.sessionList.innerHTML = groups
    .map((group) => {
      return `
        <section class="session-group">
          <div class="session-group-title">${escapeHtml(group.label)}</div>
          <div class="session-group-items">
            ${group.items.map((session) => renderSessionItem(session)).join("")}
          </div>
        </section>
      `;
    })
    .join("");
}

function renderSessionItem(session) {
  const active = session.id === state.currentSession?.id ? "active" : "";
  const sessionTitle = session.title || "未命名会话";
  return `
    <button class="session-item ${active}" type="button" data-session-id="${escapeHtml(session.id)}" title="${escapeHtml(sessionTitle)}">
      <span class="session-title">${escapeHtml(session.title || "未命名会话")}</span>
      <span class="session-meta">
        <span>${escapeHtml(session.status || "--")}</span>
        <span>${escapeHtml(formatUpdatedAt(session.updated_at))}</span>
      </span>
    </button>
  `;
}

function groupSessionsByAge(sessions) {
  const groups = [
    { label: "今天", items: [] },
    { label: "昨天", items: [] },
    { label: "7天内", items: [] },
    { label: "30天内", items: [] },
    { label: "更早", items: [] },
  ];

  sessions.forEach((session) => {
    const dayDiff = daysSince(session.updated_at);
    if (dayDiff <= 0) {
      groups[0].items.push(session);
      return;
    }
    if (dayDiff === 1) {
      groups[1].items.push(session);
      return;
    }
    if (dayDiff <= 7) {
      groups[2].items.push(session);
      return;
    }
    if (dayDiff <= 30) {
      groups[3].items.push(session);
      return;
    }
    groups[4].items.push(session);
  });

  return groups.filter((group) => group.items.length > 0);
}

function daysSince(value) {
  if (!value) {
    return 0;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return 0;
  }
  const diff = Date.now() - date.getTime();
  return Math.max(0, Math.floor(diff / (24 * 60 * 60 * 1000)));
}

function renderWorkspace() {
  if (!state.currentSession) {
    updateBadges("等待输入", "idle");
    return;
  }

  const visibleMessages = getVisibleMessages(state.currentSession.messages);
  const latestMessage = visibleMessages.at(-1) || null;
  const route = latestMessage?.intent || "等待输入";
  const status = state.currentSession.status || "idle";
  updateBadges(route, status);
}

function renderMessages() {
  if (!elements.messageList) {
    return;
  }
  if (!state.currentSession) {
    elements.messageList.innerHTML = "";
    normalizeInlineDebugLabels();
    return;
  }

  const visibleMessages = getVisibleMessages(state.currentSession.messages);
  if (visibleMessages.length === 0) {
    elements.messageList.innerHTML = "";
    normalizeInlineDebugLabels();
    return;
  }

  elements.messageList.innerHTML = visibleMessages.map((message) => renderMessageCard(message)).join("");
  normalizeInlineDebugLabels();
}

function normalizeInlineDebugLabels() {
  const titles = document.querySelectorAll(".message-inline-toggle[data-panel-type='debug'] .message-inline-heading h4");
  titles.forEach((titleElement) => {
    titleElement.textContent = "调试信息";
  });
}

function renderMessageCard(message) {
  const classes = [
    "message-card",
    message.role === "user" ? "user" : "",
    message.intent === "out_of_scope" ? "out-of-scope" : "",
    message.id === state.selectedMessageId ? "selected" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const roleLabel = message.role === "user" ? "用户输入" : "系统输出";
  const roleDotClass = message.role === "user" ? "user" : "assistant";

  return `
    <article class="${classes}" data-message-id="${escapeHtml(message.id)}">
      <div class="message-header">
        <div class="message-role">
          <span class="role-dot ${roleDotClass}"></span>
          <span>${escapeHtml(roleLabel)}</span>
        </div>
        <div class="message-meta">${escapeHtml(formatUpdatedAt(message.created_at))}</div>
      </div>
      <div class="${message.role === "assistant" ? "message-content markdown-rendered" : "message-content plain-text"}">${renderMessageContent(message)}</div>
      ${buildMessageActions(message)}
      ${buildMessageInlinePanels(message)}
    </article>
  `;
}

function renderMessageContent(message) {
  const content = String(message?.content || "");
  if (!content.trim()) {
    return "--";
  }
  if (message?.role === "assistant") {
    return renderMarkdownToHtml(content);
  }
  return formatMultiline(content);
}

function buildMessageTags(message) {
  if (message.role !== "assistant") {
    return "";
  }

  const items = [];
  if (message.intent) {
    items.push(renderTagBlock("路由", message.intent));
  }
  if (message.analysis?.module) {
    items.push(renderTagBlock("模块", message.analysis.module));
  }
  if (message.status) {
    items.push(renderTagBlock("状态", message.status));
  }
  if (message.debug?.latency_ms) {
    items.push(renderTagBlock("耗时", `${message.debug.latency_ms} ms`));
  }

  if (items.length === 0) {
    return "";
  }
  return `<div class="message-tags">${items.join("")}</div>`;
}

function buildMessageActions(message) {
  if (!Array.isArray(message.actions) || message.actions.length === 0) {
    return "";
  }
  return "";
}

function buildMessageInlinePanels(message) {
  if (message.role !== "assistant") {
    return "";
  }
  if (isAssistantResponsePending(message)) {
    return "";
  }

  const citationPanel = buildInlineCitationPanel(message);
  const debugPanel = buildInlineDebugPanel(message);

  if (!citationPanel && !debugPanel) {
    return "";
  }

  return `
    <div class="message-inline-panels">
      ${citationPanel}
      ${debugPanel}
    </div>
  `;
}

function isAssistantResponsePending(message) {
  if (!message || message.role !== "assistant") {
    return false;
  }
  const status = String(message.status || "").toLowerCase();
  const generationMode = String(message.analysis?.generation_mode || "").toLowerCase();
  const content = String(message.content || "").trim();
  if (status === "processing" || status === "pending") {
    return true;
  }
  if (generationMode === "pending") {
    return true;
  }
  return content === ASSISTANT_THINKING_TEXT;
}

function buildInlineCitationPanel(message) {
  const citations = Array.isArray(message.citations) ? message.citations : [];
  const expanded = isInlinePanelExpanded(message.id, "citation");
  const collapsedClass = expanded ? "" : " is-collapsed";
  const content =
    citations.length > 0
      ? (() => {
          const selectedIndex = getInlineCitationSelectionIndex(message.id, citations.length);
          const selectedCitation = citations[selectedIndex];
          const tabs = citations
            .map((citation, index) => {
              const activeClass = index === selectedIndex ? "active" : "";
              const title = citation?.title ? ` title="${escapeHtml(citation.title)}"` : "";
              return `
                <button
                  class="inline-citation-tab ${activeClass}"
                  type="button"
                  data-action="select-inline-citation"
                  data-message-id="${escapeHtml(message.id)}"
                  data-citation-index="${index}"
                  ${title}
                >
                  证据${index + 1}
                </button>
              `;
            })
            .join("");

          return `
            <div class="inline-citation-tabs" role="tablist" aria-label="引用证据切换">
              ${tabs}
            </div>
            <div class="inline-citation-content">
              ${renderCitationCard(selectedCitation)}
            </div>
          `;
        })()
      : `<div class="message-inline-empty">当前消息没有引用证据。</div>`;

  return `
    <section class="message-inline-panel${collapsedClass}">
      <button
        class="message-inline-panel-header message-inline-toggle"
        type="button"
        data-action="toggle-inline-panel"
        data-message-id="${escapeHtml(message.id)}"
        data-panel-type="citation"
        aria-expanded="${expanded ? "true" : "false"}"
      >
        <span class="message-inline-heading">
        <h4>引用证据</h4>
          <span class="panel-tag">Sources</span>
        </span>
        <span class="message-inline-chevron">${expanded ? "收起" : "展开"}</span>
      </button>
      <div class="message-inline-panel-body">${content}</div>
    </section>
  `;
}

function getInlineCitationSelectionIndex(messageId, citationCount) {
  if (!messageId || citationCount <= 0) {
    return 0;
  }
  const raw = state.inlineCitationSelection[messageId];
  if (Number.isInteger(raw) && raw >= 0 && raw < citationCount) {
    return raw;
  }
  return 0;
}

function formatLatencyMs(value) {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return `${value} ms`;
  }
  return "--";
}

function asYesNoUnknown(value) {
  if (typeof value !== "boolean") {
    return "--";
  }
  return value ? "yes" : "no";
}

function resolveLLMCallStatus(message, debug) {
  if (!state.debugVerboseEnabled) {
    return null;
  }
  const fromDebug = debug?.llm_call_status;
  if (fromDebug && typeof fromDebug === "object" && Object.keys(fromDebug).length > 0) {
    return fromDebug;
  }
  const fromAnalysis = message?.analysis?.llm_call_status;
  if (fromAnalysis && typeof fromAnalysis === "object" && Object.keys(fromAnalysis).length > 0) {
    return fromAnalysis;
  }
  return null;
}

function renderLLMDebugCards(llmCallStatus) {
  if (!llmCallStatus) {
    return "";
  }
  return [
    renderDebugCard("llm_status", llmCallStatus.status || "--"),
    renderDebugCard("llm_success", llmCallStatus.status === "success" ? "yes" : llmCallStatus.status ? "no" : "--"),
    renderDebugCard("llm_latency", formatLatencyMs(llmCallStatus.latency_ms)),
    renderDebugCard("llm_attempts", llmCallStatus.attempts ?? "--"),
    renderDebugCard("llm_request_sent", asYesNoUnknown(llmCallStatus.request_sent)),
    renderDebugCard("llm_reason", llmCallStatus.reason || "--"),
    renderDebugCard("llm_model", llmCallStatus.model || "--"),
  ].join("");
}

function buildInlineDebugPanel(message) {
  const rawDebug = message?.debug;
  const debug = rawDebug && typeof rawDebug === "object" ? rawDebug : {};
  const hasDebugPayload = Object.keys(debug).length > 0;
  const shouldRenderDebugPanel =
    hasDebugPayload || state.debugVerboseEnabled || Boolean(message.trace_id) || message.role === "assistant";
  if (!shouldRenderDebugPanel) {
    return "";
  }
  const expanded = isInlinePanelExpanded(message.id, "debug");
  const collapsedClass = expanded ? "" : " is-collapsed";
  const latencyValue = formatLatencyMs(debug.latency_ms);
  const llmCallStatus = resolveLLMCallStatus(message, debug);
  const gridItems = [
    renderDebugCard("message_id", message.id || "--"),
    renderDebugCard("trace_id", message.trace_id || "--"),
    renderDebugCard("route", debug.route || message.intent || "--"),
    renderDebugCard("status", message.status || "--"),
    renderDebugCard("domain", stringifyMetric(debug.domain_relevance)),
    renderDebugCard("latency", latencyValue),
    renderDebugCard("backend", debug.graph_backend || "--"),
    renderDebugCard("next_action", debug.next_action || "--"),
  ];
  if (llmCallStatus) {
    gridItems.push(renderDebugCard("llm_status", llmCallStatus.status || "--"));
    gridItems.push(
      renderDebugCard("llm_success", llmCallStatus.status === "success" ? "yes" : llmCallStatus.status ? "no" : "--")
    );
    gridItems.push(renderDebugCard("llm_latency", formatLatencyMs(llmCallStatus.latency_ms)));
    gridItems.push(renderDebugCard("llm_attempts", llmCallStatus.attempts ?? "--"));
    gridItems.push(renderDebugCard("llm_request_sent", asYesNoUnknown(llmCallStatus.request_sent)));
    gridItems.push(renderDebugCard("llm_reason", llmCallStatus.reason || "--"));
    gridItems.push(renderDebugCard("llm_model", llmCallStatus.model || "--"));
  }

  const hasAnyDebug =
    Boolean(message.trace_id) ||
    Boolean(message.intent) ||
    Boolean(message.status) ||
    Boolean(debug.route) ||
    Boolean(debug.graph_backend) ||
    Boolean(debug.next_action) ||
    typeof debug.latency_ms === "number" ||
    typeof debug.domain_relevance === "number" ||
    Boolean(llmCallStatus);

  const content = hasAnyDebug
    ? `
      <div class="message-inline-debug-grid">
        ${gridItems.join("")}
      </div>
      ${renderPathCard(debug.graph_path)}
    `
    : `<div class="message-inline-empty">当前消息没有调试信息。</div>`;

  return `
    <section class="message-inline-panel${collapsedClass}">
      <button
        class="message-inline-panel-header message-inline-toggle"
        type="button"
        data-action="toggle-inline-panel"
        data-message-id="${escapeHtml(message.id)}"
        data-panel-type="debug"
        aria-expanded="${expanded ? "true" : "false"}"
      >
        <span class="message-inline-heading">
        <h4>调试面板</h4>
          <span class="panel-tag">Trace</span>
        </span>
        <span class="message-inline-chevron">${expanded ? "收起" : "展开"}</span>
      </button>
      <div class="message-inline-panel-body">${content}</div>
    </section>
  `;
}

function getVisibleMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }
  return messages;
}

function resolveSelectedMessageId(session, preferredMessageId = null) {
  if (!session || !Array.isArray(session.messages)) {
    return null;
  }

  const visibleMessages = getVisibleMessages(session.messages);
  if (visibleMessages.length === 0) {
    return null;
  }

  if (preferredMessageId) {
    const preferred = visibleMessages.find((message) => message.id === preferredMessageId);
    if (preferred) {
      return preferred.id;
    }
  }

  const latestAssistant = [...visibleMessages].reverse().find((message) => message.role === "assistant");
  return latestAssistant?.id || visibleMessages.at(-1)?.id || null;
}

function updateBadges(route, status) {
  if (!elements.routeBadge || !elements.statusBadge) {
    return;
  }
  elements.routeBadge.textContent = route;
  elements.routeBadge.className = `badge ${badgeClassForRoute(route)}`;
  elements.statusBadge.textContent = status;
  elements.statusBadge.className = `badge ${badgeClassForStatus(status)}`;
}

function setPending(pending) {
  state.sending = pending;

  if (elements.sendButton) {
    elements.sendButton.disabled = pending;
    elements.sendButton.textContent = pending ? "处理中..." : "发送";
  }
  if (elements.newSessionBtn) {
    elements.newSessionBtn.disabled = pending;
  }
  if (elements.newSessionMiniBtn) {
    elements.newSessionMiniBtn.disabled = pending;
  }
  if (elements.sidebarToggleBtn) {
    elements.sidebarToggleBtn.disabled = pending;
  }
  if (elements.sidebarToggleMiniBtn) {
    elements.sidebarToggleMiniBtn.disabled = pending;
  }
}

function scrollMessagesToBottom() {
  if (!elements.messageList && !elements.contentGrid) {
    return;
  }
  requestAnimationFrame(() => {
    const messageList = elements.messageList;
    const contentGrid = elements.contentGrid;
    const canScrollMessageList =
      !!messageList &&
      window.getComputedStyle(messageList).overflowY !== "visible" &&
      messageList.scrollHeight > messageList.clientHeight + 1;

    if (canScrollMessageList && messageList) {
      messageList.scrollTop = messageList.scrollHeight;
      return;
    }

    if (contentGrid) {
      contentGrid.scrollTop = contentGrid.scrollHeight;
    }
  });
}

function handleError(error) {
  updateBadges("error", "error");
  console.error(error);
}

function detectCitationLanguage(citation) {
  const path = String(citation?.path || "").toLowerCase();
  const title = String(citation?.title || "").toLowerCase();
  const hint = `${path} ${title}`;
  if (hint.includes(".py")) {
    return "python";
  }
  if (hint.includes(".ts") || hint.includes(".tsx")) {
    return "typescript";
  }
  if (hint.includes(".js") || hint.includes(".jsx")) {
    return "javascript";
  }
  if (hint.includes(".json")) {
    return "json";
  }
  if (hint.includes(".yaml") || hint.includes(".yml")) {
    return "yaml";
  }
  if (hint.includes(".sql")) {
    return "sql";
  }
  if (hint.includes(".ps1")) {
    return "powershell";
  }
  if (hint.includes(".sh") || hint.includes(".bash")) {
    return "bash";
  }
  if (hint.includes(".md")) {
    return "markdown";
  }
  return "text";
}

function languageLabel(language) {
  const mapping = {
    python: "Python",
    javascript: "JavaScript",
    typescript: "TypeScript",
    json: "JSON",
    yaml: "YAML",
    sql: "SQL",
    bash: "Bash",
    powershell: "PowerShell",
    markdown: "Markdown",
    text: "Text",
  };
  return mapping[language] || "Text";
}

function sanitizeLanguageClass(language) {
  const normalized = String(language || "text").toLowerCase();
  const safe = normalized.replace(/[^a-z0-9_-]/g, "");
  return safe || "text";
}

function highlightByRegex(code, regex, tokenClassByGroupIndex) {
  let result = "";
  let lastIndex = 0;
  regex.lastIndex = 0;
  while (true) {
    const matched = regex.exec(code);
    if (!matched) {
      break;
    }
    result += escapeHtml(code.slice(lastIndex, matched.index));
    let tokenClass = "plain";
    let tokenText = matched[0];
    for (let groupIndex = 1; groupIndex < matched.length; groupIndex += 1) {
      if (matched[groupIndex] === undefined) {
        continue;
      }
      tokenClass = tokenClassByGroupIndex[groupIndex] || "plain";
      tokenText = matched[groupIndex];
      break;
    }
    result += `<span class="code-token ${tokenClass}">${escapeHtml(tokenText)}</span>`;
    lastIndex = matched.index + tokenText.length;
    if (regex.lastIndex <= matched.index) {
      regex.lastIndex = matched.index + 1;
    }
  }
  result += escapeHtml(code.slice(lastIndex));
  return result;
}

function highlightCodeByLanguage(code, language) {
  const raw = String(code || "");
  switch (language) {
    case "python":
      return highlightByRegex(
        raw,
        /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(?:def|class|return|if|elif|else|for|while|try|except|finally|with|as|import|from|pass|break|continue|and|or|not|in|is|none|true|false|lambda|yield|async|await|raise|assert)\b|\b\d+(?:\.\d+)?\b|(@[A-Za-z_][A-Za-z0-9_]*)/gim,
        {
          1: "comment",
          2: "string",
          3: "keyword",
          4: "number",
          5: "decorator",
        }
      );
    case "javascript":
    case "typescript":
      return highlightByRegex(
        raw,
        /(\/\/.*$|\/\*[\s\S]*?\*\/)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|\b(?:function|return|if|else|for|while|try|catch|finally|const|let|var|class|new|import|from|export|default|async|await|throw|switch|case|break|continue|extends|implements|interface|type)\b|\b\d+(?:\.\d+)?\b/gim,
        {
          1: "comment",
          2: "string",
          3: "keyword",
          4: "number",
        }
      );
    case "json":
      return highlightByRegex(
        raw,
        /("(?:\\.|[^"\\])*"(?=\s*:))|("(?:\\.|[^"\\])*")|\b(?:true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
        {
          1: "key",
          2: "string",
          3: "keyword",
          4: "number",
        }
      );
    case "sql":
      return highlightByRegex(
        raw,
        /(--.*$|\/\*[\s\S]*?\*\/)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(?:select|from|where|and|or|join|left|right|inner|outer|on|group|by|order|limit|insert|into|values|update|set|delete|create|table|alter|drop|as|having|union|all)\b|\b\d+(?:\.\d+)?\b/gim,
        {
          1: "comment",
          2: "string",
          3: "keyword",
          4: "number",
        }
      );
    case "bash":
    case "powershell":
      return highlightByRegex(
        raw,
        /(#.*$)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\$[A-Za-z_][A-Za-z0-9_]*|\$\{[^}]+\})|\b(?:if|then|else|fi|for|while|do|done|function|return|case|esac|in|param)\b|\b\d+(?:\.\d+)?\b/gim,
        {
          1: "comment",
          2: "string",
          3: "variable",
          4: "keyword",
          5: "number",
        }
      );
    default:
      return escapeHtml(raw);
  }
}

function renderCitationCard(citation) {
  if (!citation) {
    return "";
  }
  const sourceType = String(citation.source_type || "source");
  const symbolName = String(citation.symbol_name || "").trim();
  const startLine = Number.isInteger(citation.start_line) ? citation.start_line : null;
  const endLine = Number.isInteger(citation.end_line) ? citation.end_line : null;
  const lineMeta =
    startLine !== null
      ? `L${startLine}${endLine !== null && endLine !== startLine ? `-L${endLine}` : ""}`
      : "";
  const extraMetaItems = [
    symbolName ? `<span class="citation-meta-item">symbol ${escapeHtml(symbolName)}</span>` : "",
    lineMeta ? `<span class="citation-meta-item">${escapeHtml(lineMeta)}</span>` : "",
  ]
    .filter(Boolean)
    .join("");
  const excerptHtml = renderCitationExcerpt(citation);
  return `
    <div class="citation-card">
      <div class="citation-header">
        <span class="source-chip ${escapeHtml(sourceType)}">${escapeHtml(sourceType)}</span>
        <span class="citation-meta">score ${stringifyMetric(citation.score)}</span>
      </div>
      <div class="citation-title">${escapeHtml(citation.title || citation.path || "未命名来源")}</div>
      <p class="citation-meta">${escapeHtml(citation.path || "--")}</p>
      ${extraMetaItems ? `<div class="citation-extra-meta">${extraMetaItems}</div>` : ""}
      ${excerptHtml}
    </div>
  `;
}

function isCodeCitation(citation) {
  if (!citation) {
    return false;
  }
  const sourceType = String(citation.source_type || "").toLowerCase();
  if (sourceType.startsWith("wiki")) {
    return false;
  }
  if (sourceType.startsWith("code")) {
    return true;
  }
  if (citation.symbol_name) {
    return true;
  }
  return Number.isInteger(citation.start_line) || Number.isInteger(citation.end_line);
}

function isWikiCitation(citation) {
  if (!citation) {
    return false;
  }
  const sourceType = String(citation.source_type || "").toLowerCase();
  return sourceType.startsWith("wiki");
}

function sanitizeHttpUrl(url) {
  const raw = String(url || "").trim();
  if (/^https?:\/\//i.test(raw)) {
    return raw;
  }
  return "";
}

function renderInlineMarkdown(text) {
  let html = escapeHtml(String(text || ""));
  const inlineCodeSegments = [];
  html = html.replace(/`([^`\n]+)`/g, (_matched, codeText) => {
    const token = `__MD_INLINE_CODE_${inlineCodeSegments.length}__`;
    inlineCodeSegments.push(`<code class="md-inline-code">${codeText}</code>`);
    return token;
  });
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m, label, url) => {
    const safeUrl = sanitizeHttpUrl(url);
    if (!safeUrl) {
      return `${label}(${url})`;
    }
    return `<a class="md-link" href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^A-Za-z0-9_])\*(\S(?:[^*\n]*\S)?)\*(?![A-Za-z0-9_])/g, "$1<em>$2</em>");
  html = html.replace(/__MD_INLINE_CODE_(\d+)__/g, (_matched, indexText) => {
    const index = Number(indexText);
    if (!Number.isInteger(index) || index < 0 || index >= inlineCodeSegments.length) {
      return "";
    }
    return inlineCodeSegments[index];
  });
  return html;
}

function renderMarkdownToHtml(markdown) {
  const lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let paragraphLines = [];
  let listType = "";
  let listItems = [];
  let quoteLines = [];
  let inCodeFence = false;
  let codeLang = "text";
  let codeLines = [];

  const flushParagraph = () => {
    if (paragraphLines.length === 0) {
      return;
    }
    const content = paragraphLines.map((line) => renderInlineMarkdown(line)).join("<br />");
    blocks.push(`<p>${content}</p>`);
    paragraphLines = [];
  };

  const flushList = () => {
    if (!listType || listItems.length === 0) {
      listType = "";
      listItems = [];
      return;
    }
    blocks.push(`<${listType}>${listItems.join("")}</${listType}>`);
    listType = "";
    listItems = [];
  };

  const flushQuote = () => {
    if (quoteLines.length === 0) {
      return;
    }
    const content = quoteLines.map((line) => renderInlineMarkdown(line)).join("<br />");
    blocks.push(`<blockquote>${content}</blockquote>`);
    quoteLines = [];
  };

  const flushCodeFence = () => {
    const language = sanitizeLanguageClass(codeLang || "text");
    const highlighted = highlightCodeByLanguage(codeLines.join("\n"), language);
    blocks.push(`
      <div class="citation-excerpt-code-wrap md-code-wrap">
        <div class="citation-code-toolbar">
          <span class="citation-code-lang">${escapeHtml(languageLabel(language))}</span>
          <button class="citation-copy-btn" type="button" data-action="copy-citation-code">复制代码</button>
        </div>
        <pre class="citation-excerpt-code language-${language}"><code>${highlighted}</code></pre>
      </div>
    `);
    codeLang = "text";
    codeLines = [];
  };

  for (const rawLine of lines) {
    const line = rawLine ?? "";
    const trimmed = line.trim();

    const fenceMatched = trimmed.match(/^```([A-Za-z0-9_-]+)?$/);
    if (fenceMatched) {
      flushParagraph();
      flushList();
      flushQuote();
      if (!inCodeFence) {
        inCodeFence = true;
        codeLang = fenceMatched[1] ? fenceMatched[1].toLowerCase() : "text";
        codeLines = [];
      } else {
        inCodeFence = false;
        flushCodeFence();
      }
      continue;
    }

    if (inCodeFence) {
      codeLines.push(line);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      flushQuote();
      continue;
    }

    const headingMatched = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatched) {
      flushParagraph();
      flushList();
      flushQuote();
      const level = headingMatched[1].length;
      blocks.push(`<h${level} class="md-h md-h${level}">${renderInlineMarkdown(headingMatched[2])}</h${level}>`);
      continue;
    }

    if (/^([-*_])\1{2,}$/.test(trimmed)) {
      flushParagraph();
      flushList();
      flushQuote();
      blocks.push('<hr class="md-hr" />');
      continue;
    }

    const quoteMatched = line.match(/^>\s?(.*)$/);
    if (quoteMatched) {
      flushParagraph();
      flushList();
      quoteLines.push(quoteMatched[1]);
      continue;
    }
    flushQuote();

    const unorderedMatched = line.match(/^\s*[-*+]\s+(.+)$/);
    if (unorderedMatched) {
      flushParagraph();
      if (listType && listType !== "ul") {
        flushList();
      }
      listType = "ul";
      listItems.push(`<li>${renderInlineMarkdown(unorderedMatched[1])}</li>`);
      continue;
    }

    const orderedMatched = line.match(/^\s*\d+\.\s+(.+)$/);
    if (orderedMatched) {
      flushParagraph();
      if (listType && listType !== "ol") {
        flushList();
      }
      listType = "ol";
      listItems.push(`<li>${renderInlineMarkdown(orderedMatched[1])}</li>`);
      continue;
    }

    flushList();
    paragraphLines.push(line);
  }

  if (inCodeFence) {
    flushCodeFence();
  }
  flushParagraph();
  flushList();
  flushQuote();

  return blocks.join("");
}

function renderCitationExcerpt(citation) {
  const excerpt = String(citation?.excerpt || "");
  if (!excerpt.trim()) {
    return `<div class="citation-excerpt-text">--</div>`;
  }
  if (isCodeCitation(citation)) {
    const language = detectCitationLanguage(citation);
    const languageClass = sanitizeLanguageClass(language);
    const highlighted = highlightCodeByLanguage(excerpt, language);
    return `
      <div class="citation-excerpt-code-wrap">
        <div class="citation-code-toolbar">
          <span class="citation-code-lang">${escapeHtml(languageLabel(language))}</span>
          <button class="citation-copy-btn" type="button" data-action="copy-citation-code">复制代码</button>
        </div>
        <pre class="citation-excerpt-code language-${languageClass}"><code>${highlighted}</code></pre>
      </div>
    `;
  }
  if (isWikiCitation(citation)) {
    return `
      <div class="citation-markdown-toolbar">
        <button
          class="citation-copy-btn citation-copy-btn-wiki"
          type="button"
          data-action="copy-citation-text"
          data-copy-text="${escapeHtml(excerpt)}"
        >
          复制内容
        </button>
      </div>
      <div class="citation-excerpt-markdown markdown-rendered">${renderMarkdownToHtml(excerpt)}</div>
    `;
  }
  return `<div class="citation-excerpt-text">${formatMultiline(excerpt)}</div>`;
}

function renderDebugCard(label, value) {
  const normalizedValue = value === null || value === undefined || String(value).trim() === "" ? "--" : String(value);
  const isTooLong = normalizedValue.length > INLINE_DEBUG_VALUE_MAX_CHARS;
  const displayValue = isTooLong
    ? `${normalizedValue.slice(0, INLINE_DEBUG_VALUE_MAX_CHARS)}...`
    : normalizedValue;
  const titleAttr = isTooLong ? ` title="${escapeHtml(normalizedValue)}"` : "";
  return `
    <div class="debug-card">
      <div class="debug-label">${escapeHtml(label)}</div>
      <div class="debug-value debug-value-ellipsis"${titleAttr}>${escapeHtml(displayValue)}</div>
    </div>
  `;
}

function renderPathCard(graphPath) {
  if (!Array.isArray(graphPath) || graphPath.length === 0) {
    return "";
  }
  return `
    <div class="list-card">
      <div class="citation-title">graph_path</div>
      <ul>
        ${graphPath.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderTagBlock(label, value) {
  return `
    <div class="tag-block">
      <span class="tag-label">${escapeHtml(label)}</span>
      <span class="tag-value">${escapeHtml(value)}</span>
    </div>
  `;
}

function renderEmptyChat() {
  return `
    <div class="empty-chat">
      <div>
        <p class="eyebrow">READY FOR KNOWLEDGE QA</p>
        <h3>业务知识问答统一入口</h3>
        <p class="empty-state">
          点击“新建会话”后可直接输入问题；发送首条消息时系统会自动创建会话并进入问答流程。
        </p>
      </div>
    </div>
  `;
}

function badgeClassForRoute(route) {
  if (route === "issue_analysis") {
    return "warn";
  }
  if (route === "knowledge_qa") {
    return "success";
  }
  if (route === "out_of_scope" || route === "error") {
    return "danger";
  }
  return "neutral";
}

function badgeClassForStatus(status) {
  if (status === "completed") {
    return "success";
  }
  if (status === "out_of_scope" || status === "error") {
    return "danger";
  }
  return "neutral";
}

function formatUpdatedAt(value) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function stringifyMetric(value) {
  if (typeof value === "number") {
    return value.toFixed(2);
  }
  return value || "--";
}

function formatMultiline(value) {
  return escapeHtml(value).replace(/\n/g, "<br />");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
