# Clawd Codex MVP Implementation Plan

## 目标
构建一个可工作的 CLI 工具，支持多 LLM API（Anthropic/OpenAI/GLM），交互式对话，发布到 PyPI。

---

## Phase 1: 项目基础设施 (Foundation)

### 任务清单

#### 1.1 创建 PyPI 发布配置
- [ ] 创建 `setup.py` 或 `pyproject.toml`
- [ ] 配置 CLI 入口点：`clawd` 命令
- [ ] 添加项目元数据（名称、版本、描述）
- [ ] 配置依赖项

**依赖项**：
```python
# 核心依赖
anthropic>=0.18.0      # Anthropic SDK
openai>=1.0.0          # OpenAI SDK
zhipuai>=2.0.0         # GLM SDK (智谱)
python-dotenv>=1.0.0   # 环境变量管理
rich>=13.0.0           # 终端美化输出
prompt-toolkit>=3.0.0  # 交互式 REPL
```

#### 1.2 创建配置系统
- [ ] 设计配置文件结构：`~/.clawd/config.json`
- [ ] 创建配置管理模块：`src/config.py`
- [ ] 支持多 API 配置（Anthropic/OpenAI/GLM/自定义）
- [ ] 实现配置加密存储（API Key 安全）

**配置文件结构**：
```json
{
  "default_provider": "glm",
  "providers": {
    "anthropic": {
      "api_key": "encrypted_key",
      "base_url": "https://api.anthropic.com",
      "default_model": "claude-sonnet-4-20250514"
    },
    "openai": {
      "api_key": "encrypted_key",
      "base_url": "https://api.openai.com/v1",
      "default_model": "gpt-4"
    },
    "glm": {
      "api_key": "encrypted_key",
      "base_url": "https://open.bigmodel.cn/api/paas/v4",
      "default_model": "glm-4"
    }
  }
}
```

#### 1.3 创建 CLI 入口点
- [ ] 创建 `src/cli.py` 作为主入口
- [ ] 实现 `clawd login` 命令（交互式配置 API）
- [ ] 实现 `clawd --version` 快速响应
- [ ] 实现 `clawd --help` 帮助信息

**Git 提交点**: Phase 1 完成后 commit

---

## Phase 2: LLM API 集成 (Core Engine)

### 任务清单

#### 2.1 创建统一 API 抽象层
- [ ] 设计 LLM Provider 接口：`src/providers/base.py`
- [ ] 实现 Anthropic Provider：`src/providers/anthropic_provider.py`
- [ ] 实现 OpenAI Provider：`src/providers/openai_provider.py`
- [ ] 实现 GLM Provider：`src/providers/glm_provider.py`

**Provider 接口**：
```python
from abc import ABC, abstractmethod
from typing import Generator

class BaseProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], model: str = None) -> str:
        """同步对话"""
        pass

    @abstractmethod
    def chat_stream(self, messages: list[dict], model: str = None) -> Generator[str, None, None]:
        """流式对话"""
        pass
```

#### 2.2 集成测试
- [ ] 使用 GLM API 进行真实测试
- [ ] 测试环境变量配置：`GLM_API_KEY`
- [ ] 验证三个 Provider 都能正常工作

**Git 提交点**: Phase 2 完成后 commit

---

## Phase 3: 交互式 REPL (User Interface)

### 任务清单

#### 3.1 实现交互式 REPL
- [ ] 使用 `prompt-toolkit` 创建 REPL
- [ ] 实现多行输入支持
- [ ] 实现历史记录（上下键）
- [ ] 实现自动补全（命令、文件路径）

#### 3.2 实现对话循环
- [ ] 用户输入 → LLM API → 流式输出
- [ ] 实现 Agent Loop（工具调用循环）
- [ ] 实现会话管理（保存/加载）

#### 3.3 UI 美化
- [ ] 使用 `rich` 库美化输出
- [ ] 实现 Markdown 渲染
- [ ] 实现代码高亮
- [ ] 实现进度指示器

**Git 提交点**: Phase 3 完成后 commit

---

## Phase 4: 命令兼容层 (Command Layer)

### 任务清单

#### 4.1 实现基础命令
- [ ] `/help` - 显示帮助
- [ ] `/config` - 配置管理
- [ ] `/clear` - 清空对话
- [ ] `/save` - 保存会话
- [ ] `/load` - 加载会话
- [ ] `/exit` - 退出 REPL

#### 4.2 实现工具调用框架
- [ ] 设计 Tool 接口
- [ ] 实现基础工具（参考 TypeScript 源码）
- [ ] 实现权限管理

**Git 提交点**: Phase 4 完成后 commit

---

## Phase 5: PyPI 发布 (Release)

### 任务清单

#### 5.1 打包准备
- [ ] 完善 `README.md`（安装、使用说明）
- [ ] 添加 `LICENSE` 文件
- [ ] 创建 `MANIFEST.in`（包含必要文件）
- [ ] 测试本地安装：`pip install -e .`

#### 5.2 测试验证
- [ ] 完整测试流程：安装 → 配置 → 对话
- [ ] 测试多 API 切换
- [ ] 测试会话持久化

#### 5.3 发布到 PyPI
- [ ] 构建：`python -m build`
- [ ] 上传：`twine upload dist/*`
- [ ] 验证：`pip install clawd-codex`

**Git 提交点**: Phase 5 完成后 tag v0.1.0

---

## 执行策略

### Git 工作流
1. **每个 Phase 开始前**：创建新分支
   ```bash
   git checkout -b feature/phase-N
   ```

2. **每个任务完成后**：小步提交
   ```bash
   git add <files>
   git commit -m "feat: implement XXX"
   ```

3. **每个 Phase 完成后**：合并到 main
   ```bash
   git checkout main
   git merge feature/phase-N
   git tag phase-N-complete
   ```

4. **关键节点**：打标签便于回滚
   ```bash
   git tag -a v0.1.0 -m "MVP release"
   ```

### 后台运行
- 使用后台 agent 执行具体实现任务
- 每个 Phase 可以独立运行
- 定期检查进度并 git commit

---

## 风险与缓解

### 风险 1: API 兼容性问题
- **缓解**: 使用适配器模式统一不同 API 接口
- **回滚**: Phase 2 标签

### 风险 2: REPL 性能问题
- **缓解**: 异步流式输出
- **回滚**: Phase 3 标签

### 风险 3: PyPI 发布失败
- **缓解**: 本地充分测试
- **回滚**: Phase 4 标签

---

## 时间估算

| Phase | 预计时间 | 累计 |
|-------|---------|------|
| Phase 1 | 2-3 小时 | 2-3h |
| Phase 2 | 3-4 小时 | 5-7h |
| Phase 3 | 4-5 小时 | 9-12h |
| Phase 4 | 3-4 小时 | 12-16h |
| Phase 5 | 1-2 小时 | 13-18h |

**总计**: 13-18 小时（分多次完成）

---

## 下一步行动

请确认：
1. ✅ 这个计划是否符合你的预期？
2. ✅ 是否需要调整优先级？
3. ✅ 我可以开始 Phase 1 了吗？

确认后，我会：
1. 创建 Git 分支 `feature/phase-1`
2. 后台运行实现任务
3. 定期提交代码
4. 完成后向你汇报
