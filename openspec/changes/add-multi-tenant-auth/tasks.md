# Tasks: 多租户用户认证

## Phase 1: 数据库迁移

### T1.1 创建数据库迁移脚本
**优先级**: P0 | **预估**: 1h

创建 `migrations/001_multi_tenant.sql`，包含：
- [x] users 表（id, username, password_hash, display_name, role, status, created_at）
- [x] sessions 表添加 user_id 列
- [x] qa_request_log 添加 user_id 列
- [x] qa_evidence_log 添加 user_id 列
- [x] qa_feedback_log 添加 user_id 列
- [x] 创建相关索引

**验收标准**:
- 迁移脚本可重复执行（幂等）
- 所有新字段允许 NULL（兼容旧数据）

---

## Phase 2: 后端认证模块

### T2.1 创建 PasswordService
**优先级**: P0 | **预估**: 0.5h

创建 `src/auth/password_service.py`：
- [x] `hash_password(password: str) -> str` - 使用 bcrypt
- [x] `verify_password(password: str, hash: str) -> bool`

**依赖**: `passlib[bcrypt]`

---

### T2.2 创建 JWTService
**优先级**: P0 | **预估**: 1h

创建 `src/auth/jwt_service.py`：
- [x] `JWTConfig` dataclass（secret, issuer, expires_hours）
- [x] `UserPayload` dataclass（user_id, username）
- [x] `issue_token(payload: UserPayload) -> str`
- [x] `verify(token: str) -> UserPayload`
- [x] 从环境变量读取配置（JWT_SECRET, JWT_EXPIRES_HOURS）

**依赖**: `PyJWT`

---

### T2.3 创建 UserContext
**优先级**: P0 | **预估**: 0.5h

创建 `src/auth/context.py`：
- [x] `UserContext` dataclass（user_id, username）
- [x] `_CTX: ContextVar[UserContext | None]`
- [x] `set_user(ctx: UserContext) -> Token`
- [x] `reset_user(token: Token) -> None`
- [x] `get_user() -> UserContext`
- [x] `get_user_id() -> str`

---

### T2.4 创建登录接口
**优先级**: P0 | **预估**: 1h

创建 `src/api/routes/auth.py`：
- [x] `POST /api/auth/login`
- [x] 请求体：`{ username, password }`
- [x] 响应体：`{ access_token, token_type, user_id, username }`
- [x] 错误处理：401 用户名密码错误

---

### T2.5 创建认证中间件
**优先级**: P0 | **预估**: 1.5h

创建 `src/api/middleware/auth.py`：
- [x] 定义公开路径列表（/api/health, /api/auth/login, /, /assets/*）
- [x] 提取 Authorization header 中的 Bearer token
- [x] 验证 JWT 并注入 UserContext
- [x] 处理验证失败（401）
- [x] 在 `main.py` 中注册中间件

---

### T2.6 添加依赖包
**优先级**: P0 | **预估**: 0.5h

更新 `requirements.txt`：
- [x] `passlib[bcrypt]`
- [x] `PyJWT`

---

## Phase 3: 数据隔离改造

### T3.1 改造 Session Store
**优先级**: P0 | **预估**: 1h

改造 `src/session/async_postgres_session_store.py`：
- [x] `list_sessions()` 添加 `WHERE user_id = ?`
- [x] `get_session()` 添加 `WHERE user_id = ?`
- [x] `save_session()` 自动填充 user_id
- [x] 从 `UserContext` 获取 user_id

---

### T3.2 改造 Observability Store
**优先级**: P1 | **预估**: 1h

改造 `src/observability/async_postgres_store.py`：
- [x] `record_turn()` 添加 user_id
- [x] `record_feedback()` 添加 user_id
- [x] 从 `UserContext` 获取 user_id

---

## Phase 4: 前端改造

### T4.1 添加登录页面 UI
**优先级**: P0 | **预估**: 1h

改造 `src/web/index.html`：
- [x] 登录遮罩层 `#authOverlay`
- [x] 登录表单（用户名、密码输入框）
- [x] 登录按钮
- [x] 错误提示区域
- [x] 样式类名

---

### T4.2 添加登录样式
**优先级**: P0 | **预估**: 0.5h

改造 `src/web/assets/styles.css`：
- [x] `.auth-overlay` 全屏遮罩样式
- [x] `.auth-card` 登录卡片样式
- [x] `.auth-form` 表单样式
- [x] `.auth-error` 错误提示样式

---

### T4.3 实现认证逻辑
**优先级**: P0 | **预估**: 2h

改造 `src/web/assets/app.js`：
- [x] 状态：`token`, `user`
- [x] `login(username, password)` 函数
- [x] `logout()` 函数
- [x] `api()` 函数注入 Authorization header
- [x] 401 响应处理（清除 token，跳转登录）
- [x] `bootstrap()` 检查 localStorage 中的 token
- [x] 绑定登录表单 submit 事件

---

## Phase 5: 运维脚本

### T5.1 创建用户管理脚本
**优先级**: P0 | **预估**: 0.5h

创建 `scripts/create_user.py`：
- [x] 命令行参数：--username, --password, --display-name, --dsn
- [x] 调用 PasswordService 哈希密码
- [x] 插入 users 表
- [x] 支持幂等（已存在则更新密码）

---

## Phase 6: 测试

### T6.1 单元测试
**优先级**: P1 | **预估**: 1h

- [ ] `tests/auth/test_password_service.py`
- [ ] `tests/auth/test_jwt_service.py`
- [ ] `tests/auth/test_context.py`

---

### T6.2 集成测试
**优先级**: P1 | **预估**: 1h

- [ ] `tests/api/test_auth_login.py` - 登录流程
- [ ] `tests/api/test_session_isolation.py` - 会话隔离
- [ ] `tests/api/test_auth_required.py` - API 认证要求

---

## 任务汇总

| ID | 任务 | 优先级 | 预估 |
|----|------|--------|------|
| T1.1 | 数据库迁移脚本 | P0 | 1h |
| T2.1 | PasswordService | P0 | 0.5h |
| T2.2 | JWTService | P0 | 1h |
| T2.3 | UserContext | P0 | 0.5h |
| T2.4 | 登录接口 | P0 | 1h |
| T2.5 | 认证中间件 | P0 | 1.5h |
| T2.6 | 添加依赖包 | P0 | 0.5h |
| T3.1 | Session Store 改造 | P0 | 1h |
| T3.2 | Observability Store 改造 | P1 | 1h |
| T4.1 | 登录页面 UI | P0 | 1h |
| T4.2 | 登录样式 | P0 | 0.5h |
| T4.3 | 认证逻辑 | P0 | 2h |
| T5.1 | 用户管理脚本 | P0 | 0.5h |
| T6.1 | 单元测试 | P1 | 1h |
| T6.2 | 集成测试 | P1 | 1h |

**总计**: ~15h（约 2-3 个工作日）

---

## 执行顺序

```
Day 1:
  T1.1 → T2.6 → T2.1 → T2.2 → T2.3 → T2.4 → T2.5

Day 2:
  T3.1 → T3.2 → T5.1 → T4.1 → T4.2 → T4.3

Day 3:
  T6.1 → T6.2 → 集成测试 → 部署
```
