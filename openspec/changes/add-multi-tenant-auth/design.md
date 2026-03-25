# Design: 多租户用户认证

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              架构总览                                    │
└─────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │   HTTP Request   │
                         │ Authorization:   │
                         │ Bearer eyJ...    │
                         └────────┬─────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API Layer                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   AuthMiddleware                                                        │
│   ├── 验证 JWT 签名/过期                                                │
│   ├── 提取 user_id, username                                            │
│   └── 注入 UserContext (contextvars)                                    │
│                                                                         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         │                                                       │
         ▼                                                       ▼
┌─────────────────────────────┐           ┌─────────────────────────────┐
│       用户隔离数据          │           │       共享资源              │
│       (需改造)              │           │       (无需改造)            │
├─────────────────────────────┤           ├─────────────────────────────┤
│                             │           │                             │
│  sessions                   │           │  WikiRetriever              │
│  ├── user_id (新增)         │           │  CodeRetriever              │
│  └── WHERE user_id = ?      │           │  MCPClient                 │
│                             │           │  DomainProfile              │
│  qa_request_log             │           │  Embedding 模型             │
│  ├── user_id (新增)         │           │  Reranker 模型              │
│  └── WHERE user_id = ?      │           │                             │
│                             │           │  (单例共享，无需改造)       │
│  qa_evidence_log            │           │                             │
│  qa_feedback_log            │           │                             │
│                             │           │                             │
└─────────────────────────────┘           └─────────────────────────────┘
```

## 2. 认证流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          登录流程                                       │
└─────────────────────────────────────────────────────────────────────────┘

  前端                                后端
    │                                   │
    │  POST /api/auth/login             │
    │  { "username": "zhangsan",        │
    │    "password": "xxx" }            │
    │──────────────────────────────────►│
    │                                   │
    │                    ┌──────────────┴──────────────┐
    │                    │ 1. 查询 users 表            │
    │                    │ 2. 验证密码 (bcrypt)        │
    │                    │ 3. 签发 JWT (24h 有效)      │
    │                    └──────────────┬──────────────┘
    │                                   │
    │  { "access_token": "eyJ...",      │
    │    "user_id": "zhangsan",         │
    │    "username": "zhangsan" }       │
    │◄──────────────────────────────────│
    │                                   │
    │  localStorage.setItem(token)      │
    │                                   │


┌─────────────────────────────────────────────────────────────────────────┐
│                          请求流程                                       │
└─────────────────────────────────────────────────────────────────────────┘

  前端                                后端
    │                                   │
    │  GET /api/sessions                │
    │  Authorization: Bearer eyJ...     │
    │──────────────────────────────────►│
    │                                   │
    │                    ┌──────────────┴──────────────┐
    │                    │ AuthMiddleware:             │
    │                    │ 1. 验证 JWT                 │
    │                    │ 2. 提取 user_id             │
    │                    │ 3. set_user(UserContext)    │
    │                    ├─────────────────────────────┤
    │                    │ Handler:                    │
    │                    │ 4. get_user_id()            │
    │                    │ 5. WHERE user_id = ?        │
    │                    └──────────────┬──────────────┘
    │                                   │
    │  [{ "id": "xxx", ... }]           │
    │◄──────────────────────────────────│
    │                                   │
```

## 3. 数据模型

### 3.1 数据库 Schema

```sql
-- 用户表 (新增)
CREATE TABLE users (
    id            VARCHAR(64) PRIMARY KEY,
    username      VARCHAR(128) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    display_name  VARCHAR(128),
    role          VARCHAR(32) DEFAULT 'user',
    status        VARCHAR(32) DEFAULT 'active',
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_username ON users(username);


-- 会话表 (改造)
ALTER TABLE sessions ADD COLUMN user_id VARCHAR(64);
CREATE INDEX idx_sessions_user ON sessions(user_id);


-- 观测表 (改造)
ALTER TABLE qa_request_log ADD COLUMN user_id VARCHAR(64);
ALTER TABLE qa_evidence_log ADD COLUMN user_id VARCHAR(64);
ALTER TABLE qa_feedback_log ADD COLUMN user_id VARCHAR(64);

CREATE INDEX idx_qa_request_user ON qa_request_log(user_id);
CREATE INDEX idx_qa_evidence_user ON qa_evidence_log(user_id);
CREATE INDEX idx_qa_feedback_user ON qa_feedback_log(user_id);
```

### 3.2 数据关系

```
┌─────────────┐       ┌─────────────┐
│   users     │       │  sessions   │
├─────────────┤       ├─────────────┤
│ id (PK)     │◄──────│ user_id(FK) │
│ username    │       │ id (PK)     │
│ password    │       │ title       │
│ ...         │       │ messages    │
└─────────────┘       └─────────────┘
       │
       │
       ▼
┌─────────────────────────────────────┐
│          qa_request_log             │
├─────────────────────────────────────┤
│ user_id (FK)                        │
│ trace_id                            │
│ ...                                 │
└─────────────────────────────────────┘
```

## 4. 核心模块设计

### 4.1 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          模块依赖                                       │
└─────────────────────────────────────────────────────────────────────────┘

  src/api/main.py
       │
       ├── src/api/middleware/auth.py
       │         │
       │         └── src/auth/jwt_service.py
       │                   │
       │                   └── config (JWT_SECRET)
       │
       ├── src/api/routes/auth.py
       │         │
       │         ├── src/auth/password_service.py
       │         └── src/auth/jwt_service.py
       │
       ├── src/session/async_postgres_session_store.py
       │         │
       │         └── src/auth/context.py (get_user_id)
       │
       └── src/observability/async_postgres_store.py
                 │
                 └── src/auth/context.py (get_user_id)
```

### 4.2 目录结构

```
src/
├── auth/                           # 新增：认证模块
│   ├── __init__.py
│   ├── password_service.py         # 密码哈希/验证
│   ├── jwt_service.py              # JWT 签发/验证
│   └── context.py                  # UserContext
│
├── api/
│   ├── routes/
│   │   └── auth.py                 # 新增：认证路由
│   ├── middleware/
│   │   └── auth.py                 # 新增：认证中间件
│   └── main.py                     # 改造：注册中间件
│
├── session/
│   └── async_postgres_session_store.py  # 改造：user_id 隔离
│
└── observability/
    └── async_postgres_store.py          # 改造：user_id
```

## 5. 接口规范

### 5.1 登录接口

```yaml
POST /api/auth/login
Content-Type: application/json

Request:
  username: string (required)
  password: string (required)

Response 200:
  access_token: string   # JWT Token
  token_type: "Bearer"
  user_id: string
  username: string

Response 401:
  detail: "用户名或密码错误"
```

### 5.2 认证中间件行为

```yaml
公开路径 (无需认证):
  - /api/health
  - /api/auth/login
  - / (根路径)
  - /assets/* (静态资源)

认证路径:
  - 其他所有 /api/* 路径

请求头:
  Authorization: Bearer <jwt_token>

验证失败:
  - 缺少 Header: 401 "Missing token"
  - Token 无效: 401 "Invalid token"
  - Token 过期: 401 "Token expired"
```

## 6. 前端设计

### 6.1 登录页面

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                     ┌────────────────────────────────┐                 │
│                     │      鲸鸿动能                   │                 │
│                     │      知识问答系统               │                 │
│                     ├────────────────────────────────┤                 │
│                     │                                │                 │
│                     │  用户名                         │                 │
│                     │  ┌────────────────────────────┐│                 │
│                     │  │                            ││                 │
│                     │  └────────────────────────────┘│                 │
│                     │                                │                 │
│                     │  密码                           │                 │
│                     │  ┌────────────────────────────┐│                 │
│                     │  │ ******                     ││                 │
│                     │  └────────────────────────────┘│                 │
│                     │                                │                 │
│                     │  [        登  录        ]      │                 │
│                     │                                │                 │
│                     │  ⚠️ 用户名或密码错误           │  ← 错误提示     │
│                     │                                │                 │
│                     └────────────────────────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Token 管理

```javascript
// 存储位置
localStorage:
  - token: string          // JWT Token
  - user: { user_id, username }  // 用户信息

// 生命周期
登录成功 → 存储 token → 每次请求携带 → 401 时清除 → 跳转登录
```

## 7. 配置项

```yaml
环境变量:
  JWT_SECRET: string        # JWT 签名密钥 (必需)
  JWT_EXPIRES_HOURS: int    # Token 有效期 (默认: 24)
  JWT_ISSUER: string        # JWT 签发者 (默认: dsp-agent)
```

## 8. 运维脚本

```bash
# 创建用户
python scripts/create_user.py \
  --username zhangsan \
  --password "YourPassword123!" \
  --display-name "张三"

# 输出
✓ 用户已创建: zhangsan
  显示名称: 张三
```

## 9. 测试策略

```
单元测试:
├── auth/password_service_test.py    # 密码哈希/验证
├── auth/jwt_service_test.py         # JWT 签发/验证
└── auth/context_test.py             # ContextVar 行为

集成测试:
├── test_auth_login.py               # 登录流程
├── test_session_isolation.py        # 会话隔离
└── test_api_auth_required.py        # API 认证要求

E2E 测试:
└── test_user_flow.py                # 完整用户流程
```

## 10. 迁移策略

```
Step 1: 部署数据库迁移
        └── 创建 users 表
        └── sessions/obs 添加 user_id (允许 NULL)

Step 2: 部署后端代码
        └── 新增认证模块
        └── 改造 session/observability

Step 3: 创建初始用户
        └── 使用 scripts/create_user.py

Step 4: 部署前端
        └── 登录页面
        └── Token 管理

Step 5: 清理旧数据 (可选)
        └── 删除 user_id 为 NULL 的会话
        └── 或迁移到默认用户
```
