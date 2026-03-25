# Proposal: 多租户用户认证

## 概述

为系统添加多租户用户认证能力，支持用户名+密码登录，实现会话数据隔离。

## 背景

当前系统无认证机制，所有用户共享同一数据空间。需要：
- 区分不同用户的会话数据
- 提供基本的登录认证
- 保持共享检索资源（Wiki/Code 知识库）

## 目标

1. 用户可通过用户名+密码登录系统
2. 每个用户的会话数据相互隔离
3. 观测数据按用户隔离
4. 检索资源（Wiki/Code）保持共享

## 非目标

- 不实现共享会话（后续考虑）
- 不实现租户配置（技能、MCP 工具等保持全局共享）
- 不实现 Admin 管理后台
- 不实现用户自助注册

## 方案设计

### 用户模型

采用 **1 User = 1 Tenant** 模式：
- 每个用户独立，不共享会话
- 无并发冲突问题
- 逻辑简单

### 认证流程

```
用户名+密码 → JWT Token → API 请求携带 Token → 后端验证 → UserContext
```

### 数据隔离

- 会话（sessions）：按 `user_id` 隔离
- 观测（qa_*_log）：按 `user_id` 隔离
- 检索资源：共享，无需改造

## 涉及文件

### 后端新增

| 文件 | 说明 |
|------|------|
| `src/auth/password_service.py` | 密码哈希/验证 |
| `src/auth/jwt_service.py` | JWT 签发/验证 |
| `src/auth/context.py` | UserContext (contextvars) |
| `src/api/routes/auth.py` | POST /api/auth/login |
| `scripts/create_user.py` | 运维脚本：创建用户 |

### 后端改造

| 文件 | 改动 |
|------|------|
| `src/session/async_postgres_session_store.py` | 添加 user_id 隔离 |
| `src/observability/async_postgres_store.py` | 添加 user_id |
| `src/api/main.py` | 注册认证中间件/路由 |

### 前端改造

| 文件 | 改动 |
|------|------|
| `src/web/index.html` | 登录页 UI |
| `src/web/assets/app.js` | 认证逻辑、Token 管理 |
| `src/web/assets/styles.css` | 登录样式 |

### 数据库迁移

| 文件 | 说明 |
|------|------|
| `migrations/001_multi_tenant.sql` | users 表 + sessions/obs 添加 user_id |

## API 变更

### 新增端点

```
POST /api/auth/login
Request:  { "username": string, "password": string }
Response: { "access_token": string, "token_type": "Bearer", "user_id": string, "username": string }
```

### 现有端点变更

所有需要认证的端点：
- 请求需携带 `Authorization: Bearer <token>`
- 未认证返回 401
- 数据自动按 user_id 隔离

## 兼容性评估

- **现有 API**：需添加认证 Header，不兼容旧客户端
- **数据库**：新增字段，旧数据 user_id 为 NULL（需迁移或清理）
- **前端**：需完全替换为带认证的版本

## 风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| JWT Secret 泄露 | 所有 Token 失效 | 环境变量管理，定期轮换 |
| 密码弱 | 账户被盗 | 运维创建用户时强制密码强度 |
| Token 过期 | 用户体验中断 | 合理设置过期时间（24h） |

## 时间估算

| 阶段 | 工作量 |
|------|--------|
| 数据库 | 2 小时 |
| 后端认证 | 1 天 |
| 数据隔离 | 0.5 天 |
| 前端改造 | 1 天 |
| 测试 | 2 小时 |
| **总计** | **~3 天** |

## 依赖

- Python 包：`passlib[bcrypt]`, `PyJWT`
- 无新增外部服务

## 后续演进

1. 共享会话（多用户同一租户）
2. Admin 管理后台
3. 用户自助注册
4. SSO 集成
