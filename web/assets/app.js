// 前端运行时状态。
// 这里不引入额外状态管理库，直接用一个轻量对象维护当前会话、选中消息和发送态。
const state = {
  sessions: [],
  currentSession: null,
  selectedMessageId: null,
  sending: false,
};

// 统一缓存页面上的关键 DOM 节点，避免在每次渲染时反复 querySelector。
const elements = {
  newSessionBtn: document.getElementById("newSessionBtn"),
  sessionCountTag: document.getElementById("sessionCountTag"),
  sessionList: document.getElementById("sessionList"),
  workspaceTitle: document.getElementById("workspaceTitle"),
  workspaceSubtitle: document.getElementById("workspaceSubtitle"),
  routeBadge: document.getElementById("routeBadge"),
  statusBadge: document.getElementById("statusBadge"),
  messageList: document.getElementById("messageList"),
  composerForm: document.getElementById("composerForm"),
  composerInput: document.getElementById("composerInput"),
  sendButton: document.getElementById("sendButton"),
  analysisPanel: document.getElementById("analysisPanel"),
  citationPanel: document.getElementById("citationPanel"),
  debugPanel: document.getElementById("debugPanel"),
};

document.addEventListener("DOMContentLoaded", () => {
  // 页面加载后先绑定事件，再触发初始化拉取，保证首屏和交互都走同一套逻辑。
  bindEvents();
  bootstrap().catch(handleError);
});

function bindEvents() {
  // 新建会话按钮：直接创建一个新的后端会话。
  elements.newSessionBtn.addEventListener("click", () => {
    createSession().catch(handleError);
  });

  // 输入框提交：统一走 submitMessage，避免页面默认 form 提交刷新。
  elements.composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitMessage().catch(handleError);
  });

  // 事件委托统一处理页面中的动态节点，例如会话项、快捷提问和代码确认按钮。
  document.addEventListener("click", (event) => {
    const promptButton = event.target.closest("[data-prompt]");
    if (promptButton) {
      // 快捷提问只负责把推荐问题写入输入框，不直接发送，方便用户再改一句。
      elements.composerInput.value = promptButton.dataset.prompt || "";
      elements.composerInput.focus();
      return;
    }

    const sessionButton = event.target.closest("[data-session-id]");
    if (sessionButton) {
      openSession(sessionButton.dataset.sessionId).catch(handleError);
      return;
    }

    const actionButton = event.target.closest("[data-action='confirm-code']");
    if (actionButton) {
      confirmCodeGeneration(actionButton.dataset.messageId).catch(handleError);
      return;
    }

    const messageCard = event.target.closest(".message-card.selectable[data-message-id]");
    if (messageCard) {
      // 只有助手消息允许被选中，因为右侧面板展示的是分析、证据和 debug 信息。
      state.selectedMessageId = messageCard.dataset.messageId;
      renderSelectedPanels();
      renderMessages();
    }
  });
}

async function bootstrap() {
  // 首屏先拉会话列表，没有会话就自动创建一个，保证页面打开后可立即交互。
  elements.messageList.innerHTML = `<div class="loading-shell">正在加载会话数据...</div>`;
  await refreshSessions();
  if (state.sessions.length === 0) {
    await createSession();
    return;
  }
  await openSession(state.sessions[0].id);
}

async function api(path, options = {}) {
  // 所有接口调用统一走这里，保证错误处理和请求头策略一致。
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
      // 某些异常响应可能不是 JSON，这里静默忽略解析失败，保留默认错误文案。
    }
    throw new Error(errorText);
  }

  return response.json();
}

async function refreshSessions() {
  // 会话列表是左侧栏的数据源；每次创建会话或发送消息后都刷新一次，保持顺序正确。
  const payload = await api("/api/sessions");
  state.sessions = payload.items || [];
  renderSessionList();
}

async function createSession() {
  // 创建会话时先切 pending，避免用户重复点击。
  setPending(true);
  try {
    const payload = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({}),
    });
    await refreshSessions();
    state.currentSession = payload.session;
    state.selectedMessageId = payload.session.messages.at(-1)?.id || null;
    renderAll();
  } finally {
    setPending(false);
  }
}

async function openSession(sessionId) {
  // 打开会话时优先选中最新的一条助手消息，让右侧摘要面板立即有内容。
  if (!sessionId) {
    return;
  }

  const payload = await api(`/api/sessions/${sessionId}`);
  state.currentSession = payload.session;
  const latestAssistant = [...state.currentSession.messages]
    .reverse()
    .find((message) => message.role === "assistant");
  state.selectedMessageId = latestAssistant?.id || state.currentSession.messages.at(-1)?.id || null;
  renderAll();
}

async function submitMessage() {
  // 发送消息前做三类保护：空内容、无会话、正在发送中。
  const content = elements.composerInput.value.trim();
  if (!content || !state.currentSession || state.sending) {
    return;
  }

  setPending(true);
  try {
    const payload = await api("/api/messages", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.currentSession.id,
        content,
      }),
    });
    await refreshSessions();
    state.currentSession = payload.session;
    state.selectedMessageId = payload.assistant_message_id || state.currentSession.messages.at(-1)?.id || null;
    elements.composerInput.value = "";
    renderAll();
    scrollMessagesToBottom();
  } finally {
    setPending(false);
  }
}

async function confirmCodeGeneration(messageId) {
  // 这个动作对应后端的“从问题分析继续进入代码生成路径”。
  if (!messageId || state.sending) {
    return;
  }

  setPending(true);
  try {
    const payload = await api(`/api/messages/${messageId}/confirm-code`, {
      method: "POST",
      body: JSON.stringify({ approved: true }),
    });
    await refreshSessions();
    state.currentSession = payload.session;
    state.selectedMessageId = payload.assistant_message_id || state.currentSession.messages.at(-1)?.id || null;
    renderAll();
    scrollMessagesToBottom();
  } finally {
    setPending(false);
  }
}

function renderAll() {
  // 全量重渲染入口。当前页面规模不大，这样写更直接，后续如有性能问题再细拆。
  renderSessionList();
  renderWorkspace();
  renderMessages();
  renderSelectedPanels();
}

function renderSessionList() {
  // 左侧会话列表只展示摘要信息，不展开消息详情。
  elements.sessionCountTag.textContent = String(state.sessions.length);
  if (state.sessions.length === 0) {
    elements.sessionList.innerHTML = `<div class="empty-state">还没有会话，点击上方按钮创建一个。</div>`;
    return;
  }

  elements.sessionList.innerHTML = state.sessions
    .map((session) => {
      const active = session.id === state.currentSession?.id ? "active" : "";
      return `
        <button class="session-item ${active}" type="button" data-session-id="${escapeHtml(session.id)}">
          <span class="session-title">${escapeHtml(session.title)}</span>
          <span class="session-meta">
            <span>${escapeHtml(session.status)}</span>
            <span>${escapeHtml(formatUpdatedAt(session.updated_at))}</span>
          </span>
        </button>
      `;
    })
    .join("");
}

function renderWorkspace() {
  // 顶部标题区跟随当前会话和当前最新消息状态切换。
  if (!state.currentSession) {
    elements.workspaceTitle.textContent = "准备开始新会话";
    elements.workspaceSubtitle.textContent = "发送一个业务问题、报错描述或系统设计问题，界面会根据返回结果切换状态。";
    updateBadges("等待输入", "idle");
    return;
  }

  const latestMessage = state.currentSession.messages.at(-1);
  const route = latestMessage?.intent && latestMessage.intent !== "system" ? latestMessage.intent : "等待输入";
  const status = state.currentSession.status || "idle";
  elements.workspaceTitle.textContent = state.currentSession.title;
  elements.workspaceSubtitle.textContent =
    "左侧管理会话，中间继续多轮对话，右侧查看证据、分析摘要和 trace 调试信息。";
  updateBadges(route, status);
}

function renderMessages() {
  // 中间聊天区渲染完整消息列表，空会话时给出引导卡片。
  if (!state.currentSession) {
    elements.messageList.innerHTML = `<div class="loading-shell">正在准备会话...</div>`;
    return;
  }

  if (state.currentSession.messages.length === 0) {
    elements.messageList.innerHTML = renderEmptyChat();
    return;
  }

  elements.messageList.innerHTML = state.currentSession.messages
    .map((message) => renderMessageCard(message))
    .join("");
}

function renderMessageCard(message) {
  // 根据消息类型计算 class，驱动颜色、是否可选中等视觉效果。
  const classes = [
    "message-card",
    message.role === "user" ? "user" : "",
    message.intent === "out_of_scope" ? "out-of-scope" : "",
    message.role === "assistant" ? "selectable" : "",
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
      <div class="message-content">${formatMultiline(message.content)}</div>
      ${buildMessageTags(message)}
      ${buildMessageActions(message)}
    </article>
  `;
}

function buildMessageTags(message) {
  // 这里展示的是快速扫描信息，让用户在聊天区就能看到路由、模块和状态。
  if (message.role !== "assistant") {
    return "";
  }

  const items = [];
  if (message.intent && message.intent !== "system") {
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
  // 目前只有“需要代码实现”这一类消息动作，后续可继续扩展更多 action。
  if (!Array.isArray(message.actions) || message.actions.length === 0) {
    return "";
  }

  const buttons = message.actions
    .map((action) => {
      if (action.type !== "confirm_code_generation") {
        return "";
      }
      const disabled = state.sending ? "disabled" : "";
      return `
        <button
          class="secondary-button"
          type="button"
          data-action="confirm-code"
          data-message-id="${escapeHtml(message.id)}"
          ${disabled}
        >
          ${escapeHtml(action.label)}
        </button>
      `;
    })
    .join("");

  return buttons ? `<div class="message-actions">${buttons}</div>` : "";
}

function renderSelectedPanels() {
  // 右侧三个面板都基于“当前选中助手消息”渲染。
  const selectedMessage = getSelectedMessage();
  renderAnalysisPanel(selectedMessage);
  renderCitationPanel(selectedMessage);
  renderDebugPanel(selectedMessage);
}

function getSelectedMessage() {
  // 如果当前没有显式选中项，则回退到最新助手消息，保证右侧始终有内容。
  if (!state.currentSession) {
    return null;
  }
  return (
    state.currentSession.messages.find((message) => message.id === state.selectedMessageId) ||
    [...state.currentSession.messages].reverse().find((message) => message.role === "assistant") ||
    null
  );
}

function renderAnalysisPanel(message) {
  // 分析摘要面板只消费结构化 analysis 字段，不直接解析自然语言正文。
  if (!message?.analysis) {
    elements.analysisPanel.innerHTML = `<div class="empty-state">选中一条助手消息后，这里会显示模块、根因、修复建议和测试要点。</div>`;
    return;
  }

  const analysis = message.analysis;
  const metrics = [];
  if (analysis.module) {
    metrics.push(renderMetricCard("目标模块", analysis.module));
  }
  if (analysis.confidence) {
    metrics.push(renderMetricCard("置信度", analysis.confidence));
  }
  if (analysis.summary) {
    metrics.push(renderMetricCard("摘要", analysis.summary));
  }

  const sections = [];
  if (analysis.root_cause) {
    sections.push(renderListCard("根因判断", [analysis.root_cause]));
  }
  if (Array.isArray(analysis.fix_plan)) {
    sections.push(renderListCard("修复建议", analysis.fix_plan));
  }
  if (Array.isArray(analysis.risks)) {
    sections.push(renderListCard("风险提示", analysis.risks));
  }
  if (Array.isArray(analysis.verification_steps)) {
    sections.push(renderListCard("验证步骤", analysis.verification_steps));
  }
  if (Array.isArray(analysis.patch_summary)) {
    sections.push(renderListCard("补丁摘要", analysis.patch_summary));
  }
  if (Array.isArray(analysis.test_plan)) {
    sections.push(renderListCard("测试建议", analysis.test_plan));
  }
  if (Array.isArray(analysis.highlights)) {
    sections.push(renderListCard("补充说明", analysis.highlights));
  }
  if (Array.isArray(analysis.files)) {
    sections.push(renderListCard("涉及文件", analysis.files));
  }
  if (analysis.snippet) {
    sections.push(`
      <div class="snippet-card">
        <div class="citation-title">代码片段建议</div>
        <pre>${escapeHtml(analysis.snippet)}</pre>
      </div>
    `);
  }

  elements.analysisPanel.innerHTML = `
    <div class="inline-grid">${metrics.join("")}</div>
    <div class="analysis-list">${sections.join("")}</div>
  `;
}

function renderCitationPanel(message) {
  // 引用证据面板统一渲染 citations，来源类型可能是 wiki / case / code。
  if (!Array.isArray(message?.citations) || message.citations.length === 0) {
    elements.citationPanel.innerHTML = `<div class="empty-state">当前还没有引用证据。</div>`;
    return;
  }

  elements.citationPanel.innerHTML = `
    <div class="citation-list">
      ${message.citations.map((citation) => renderCitationCard(citation)).join("")}
    </div>
  `;
}

function renderDebugPanel(message) {
  // 调试面板直接暴露后端 trace、route、backend 和 graph_path，方便观察编排结果。
  if (!message) {
    elements.debugPanel.innerHTML = `<div class="empty-state">等待本轮 trace、路由与耗时信息。</div>`;
    return;
  }

  const debug = message.debug || {};
  elements.debugPanel.innerHTML = `
    <div class="debug-grid">
      ${renderDebugCard("message_id", message.id)}
      ${renderDebugCard("trace_id", message.trace_id || "--")}
      ${renderDebugCard("route", debug.route || message.intent || "--")}
      ${renderDebugCard("status", message.status || "--")}
      ${renderDebugCard("domain", stringifyMetric(debug.domain_relevance))}
      ${renderDebugCard("latency", debug.latency_ms ? `${debug.latency_ms} ms` : "--")}
      ${renderDebugCard("backend", debug.graph_backend || "--")}
      ${renderDebugCard("next_action", debug.next_action || "--")}
    </div>
    ${renderPathCard(debug.graph_path)}
  `;
}

function updateBadges(route, status) {
  // 顶部 badge 颜色和文本都由 route/status 推导。
  elements.routeBadge.textContent = route;
  elements.routeBadge.className = `badge ${badgeClassForRoute(route)}`;
  elements.statusBadge.textContent = status;
  elements.statusBadge.className = `badge ${badgeClassForStatus(status)}`;
}

function setPending(pending) {
  // 统一处理按钮禁用态，避免多处手工维护。
  state.sending = pending;
  elements.sendButton.disabled = pending;
  elements.newSessionBtn.disabled = pending;
  elements.sendButton.textContent = pending ? "处理中..." : "发送";
}

function scrollMessagesToBottom() {
  // 新消息渲染完成后滚动到底部，requestAnimationFrame 保证 DOM 已更新。
  requestAnimationFrame(() => {
    elements.messageList.scrollTop = elements.messageList.scrollHeight;
  });
}

function handleError(error) {
  // 当前错误只在调试面板展示，不打断整页结构。
  updateBadges("error", "error");
  elements.debugPanel.innerHTML = `
    <div class="debug-card">
      <div class="debug-label">error</div>
      <div class="debug-value">${escapeHtml(error.message || String(error))}</div>
    </div>
  `;
}

function renderMetricCard(label, value) {
  // 小型统计卡片，复用在分析摘要顶部。
  return `
    <div class="metric-card">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function renderListCard(label, items) {
  // 通用列表卡片，用于修复建议、风险、验证步骤等结构化列表。
  if (!items || items.length === 0) {
    return "";
  }

  return `
    <div class="list-card">
      <div class="citation-title">${escapeHtml(label)}</div>
      <ul>
        ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderCitationCard(citation) {
  // 单条证据卡片，展示来源类型、标题、路径和摘录。
  const sourceType = citation.source_type || "source";
  return `
    <div class="citation-card">
      <div class="citation-header">
        <span class="source-chip ${escapeHtml(sourceType)}">${escapeHtml(sourceType)}</span>
        <span class="citation-meta">score ${stringifyMetric(citation.score)}</span>
      </div>
      <div class="citation-title">${escapeHtml(citation.title || citation.path || "未命名来源")}</div>
      <p class="citation-meta">${escapeHtml(citation.path || "--")}</p>
      <p>${escapeHtml(citation.excerpt || "")}</p>
    </div>
  `;
}

function renderDebugCard(label, value) {
  // 调试字段卡片，设计成统一的小网格，便于扩展更多运行态字段。
  return `
    <div class="debug-card">
      <div class="debug-label">${escapeHtml(label)}</div>
      <div class="debug-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function renderPathCard(graphPath) {
  // graph_path 用列表直出，可以非常直观地看到后端到底走过哪些节点。
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
  // 聊天消息中的轻量标签块。
  return `
    <div class="tag-block">
      <span class="tag-label">${escapeHtml(label)}</span>
      <span class="tag-value">${escapeHtml(value)}</span>
    </div>
  `;
}

function renderEmptyChat() {
  // 空状态引导区，核心作用是告诉用户系统支持的两类主能力。
  return `
    <div class="empty-chat">
      <div>
        <p class="eyebrow">READY FOR ROUTING</p>
        <h3>从知识问答到问题分析的统一入口</h3>
        <p class="empty-state">
          你可以直接输入业务规则问题、异常现象、日志线索，或者点击左侧的快速提问示例。
        </p>
      </div>
      <div class="prompt-grid">
        <button class="prompt-chip" type="button" data-prompt="订单结算规则是什么？">先问业务知识</button>
        <button class="prompt-chip" type="button" data-prompt="库存回调重复导致解锁失败，应该怎么定位？">先做问题分析</button>
      </div>
    </div>
  `;
}

function badgeClassForRoute(route) {
  // 路由类型控制颜色：分析偏 warn，知识问答偏 success，无关和错误用 danger。
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
  // 状态 badge 单独映射，和 route 的语义不完全一样。
  if (status === "completed") {
    return "success";
  }
  if (status === "confirm_code") {
    return "warn";
  }
  if (status === "out_of_scope" || status === "error") {
    return "danger";
  }
  return "neutral";
}

function formatUpdatedAt(value) {
  // 左侧列表和消息时间统一格式化到“时:分”。
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
  // 调试数值统一保留两位小数，避免页面上出现长串浮点误差。
  if (typeof value === "number") {
    return value.toFixed(2);
  }
  return value || "--";
}

function formatMultiline(value) {
  // 聊天正文允许换行，但仍然先做 HTML 转义。
  return escapeHtml(value).replace(/\n/g, "<br />");
}

function escapeHtml(value) {
  // 所有字符串输出到 innerHTML 前都必须过这里，防止意外的 HTML 注入。
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
