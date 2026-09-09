# Subagents：子任务委派

CyberClaw 的主 Agent 现在可以通过 delegate_task 调用独立子 Agent。子任务使用自己的模型/工具循环，只返回最终结果给主 Agent；主 Agent 等待并负责汇总。它不是后台作业服务，不会在 CLI 退出后继续执行。

## CLI 用法

更新源码并重启 cyberclaw run。默认注册两种角色：

| 名称 | 作用 | 默认工具 |
| --- | --- | --- |
| code_reviewer | 阅读代码，分析问题与测试缺口 | list_office_files、read_office_file |
| document_analyst | 阅读资料、提炼信息、整理草稿 | list_office_files、read_office_file |

例如先将待分析文件放入 workspace/office/project/，再输入：

> 请让 code_reviewer 检查 project/heartbeat.py 的任务恢复逻辑，再由你汇总问题和修改建议。

或者：

> 请让 document_analyst 阅读 project/README.md，整理一份安装说明草稿。

路径相对于 office。默认角色没有写文件、Shell、画像或任务调度工具，也不会自动获得所有动态技能。它们生成建议和草稿，不能宣称已修改文件。工具列表来自可信的宿主配置；模型不能通过参数增加权限。

## 执行流程

~~~text
主 Agent → delegate_task(agent_name, task, context)
         → 注册表校验 + 并发准入
         → 新 thread_id / run_id + 独立图状态
         → 子模型 ↔ 允许的 ToolNode 工具
         → JSON 状态与最终结果
         → 主 Agent 检查状态、汇总
~~~

每次委派：

- 只收到 task 和显式 context，不复制主对话历史，不自动读取全局用户画像。
- 单独构造状态图，compile(checkpointer=False)，不继承父 checkpoint 或内部运行键。
- 新建不可由模型填写的子 thread_id 和 run_id。主会话的技能 help_token 不能用于子任务。
- 只绑定注册角色的工具。不提供 delegate_task，且运行配置 subagent_depth=1 会拒绝进一步委派。
- 同步主图和异步主图都可调用；已有事件循环中直接调用工具时请使用 ainvoke。
- 完成、失败或超时后不保存可恢复的子图状态；再次委派从新上下文开始。

这些是上下文与工具注册隔离，不是文件系统、进程、网络或多租户隔离。默认角色可以读取 office 内允许的文件；没有按任务单独划分文件目录。

## 预算与返回值

默认每个主 app 的 runner 最多同时接收 2 个子图调用；满额立即返回 busy，没有额外排队。单次等待预算为 120 秒、最多 16 个图步骤（模型节点和工具节点均计入），结果最多 6000 字符。task 最多 8000 字符，context 最多 16000 字符；超长或额外字段在 schema 层拒绝。

返回 JSON 字段：

~~~json
{
  "status": "completed",
  "agent_name": "code_reviewer",
  "run_id": "<child-run-id>",
  "parent_run_id": "<parent-run-id>",
  "result": "结论、证据和未解决问题",
  "truncated": false
}
~~~

status 可能为 completed、rejected、busy、timeout、step_limit 或 failed。失败不会伪装成成功；未知角色、缺少 thread_id 和递归委派会被拒绝。异常只返回类别，不向模型暴露提供商异常中的潜在敏感内容。

取消外层调用时，CancelledError 继续向上传播，并记录 subagent_cancelled。

超时使用协作式取消：已启动的同步工具、线程、Shell 子进程或远端请求不保证立即停止。并发上限限制的是被 runner 接收的图调用，不是对残留线程或远端计算的 OS 配额。为自定义副作用工具另行设计幂等性、超时与可终止执行环境；图步数也不是 token 或费用预算。

## SDK：默认或自定义角色

默认 create_agent_app() 会注册 delegate_task，沿用主模型提供商和模型名。显式传 tools=[...] 的旧调用默认保持原工具集合；设置 enable_subagents=True 或提供 subagent_specs 可启用。enable_subagents=False 总是关闭自动注册。

~~~python
from uuid import uuid4
from langchain_core.messages import HumanMessage
from cyberclaw.core.agent import create_agent_app
from cyberclaw.core.subagents import SubagentSpec
from cyberclaw.core.tools.sandbox_tools import read_office_file

reviewer = SubagentSpec(
    name="reviewer",
    description="阅读指定文件，评审实现并提供依据。",
    instructions="只分析当前任务。区分代码事实、推测与改进建议。",
    tools=(read_office_file,),
    # 可选 provider_name/model_name；省略时继承主 app 选择。
)
app = create_agent_app(
    provider_name="openai",
    model_name="gpt-4o-mini",
    subagent_specs=[reviewer],
)
config = {"configurable": {"thread_id": "my-session", "run_id": uuid4().hex}}
result = app.invoke(
    {"messages": [HumanMessage(content="请委派 reviewer 阅读 project/agent.py 并给出评审。")]},
    config=config,
)
print(result["messages"][-1].content)
~~~

自定义角色的 tools 是显式授权集合。角色名称必须唯一，工具名称不能重复，也不能包含 delegate_task。如需技能，在宿主代码中选择具体的动态技能加入 tools；子 Agent 仍须在自己的执行作用域中 help → run。不要把整份 BUILTIN_TOOLS 或所有技能自动作为子 Agent 的默认权限。

## SDK：显式预算与直接调用

~~~python
import asyncio
from uuid import uuid4
from cyberclaw.core.subagents import SubagentRunner

runner = SubagentRunner(
    provider_name="openai",
    model_name="gpt-4o-mini",
    max_concurrent=2,
    timeout_seconds=90,
    max_steps=12,
    max_result_chars=4000,
)
delegate = runner.as_tool()
config = {"configurable": {"thread_id": "sdk-session", "run_id": uuid4().hex}}

async def main():
    result = await delegate.ainvoke({
        "agent_name": "document_analyst",
        "task": "整理安装说明",
        "context": "仅阅读 project/README.md；返回草稿与缺失信息。",
    }, config=config)
    print(result)

asyncio.run(main())
~~~

同步代码可使用 delegate.invoke；在已有事件循环中直接 await delegate.ainvoke。若需要把这个自定义 runner 放入主 Agent，显式传入 tools=[delegate, ...]，并设置 enable_subagents=False，避免自动注册第二个同名委派工具。

## 审计关联

CLI 每次主任务生成新的 run_id。子事件写到父 thread_id 对应的 JSONL，方便现有 monitor 在同一文件中显示，并包含：

- run_id：当前子任务 ID。
- parent_run_id：父任务 ID。
- agent_name：注册角色名称。
- execution_thread_id：子图用于技能凭据隔离的 thread_id。

生命周期使用 system_action 的 action 字段：subagent_started、subagent_completed、subagent_timeout、subagent_step_limit、subagent_failed、subagent_cancelled、subagent_rejected、subagent_busy。子模型和工具沿用 llm_input、tool_call、tool_result、ai_message 四类事件；模型日志和返回结果对标准 help_token 字段/文本脱敏。

SDK 应在可信 config.configurable 中传 run_id，以关联主任务和子任务；未提供时 parent_run_id 为 null，仍可通过父 thread_id 与子 run_id 定位。模型 schema 只有 agent_name、task、context，不包含会话、模型、工具清单和预算。

日志仍使用有界队列与截断结果，不能保证完整回放或无丢失。没有单独实现持久化子任务列表、重启恢复、任务查询/取消 API 或后台通知。

## 验证

tests/test_subagents.py 覆盖真实子图、主图 ToolNode 的同步/异步往返、独立上下文、模型选择、只读工具、未知工具拒绝、技能凭据跨作用域拒绝、并发准入、超时/取消、步数限制、结果截断、异常脱敏和监控关联。测试不调用真实模型 API。

2026-09-09，Ubuntu-24.04 / WSL2、Python 3.12.3：117 项 unittest（含 24 项子 Agent 测试）、独立懒加载测试通过；实际 wheel 隔离构建安装后，成功导入子 Agent 模块并创建委派工具。

参考：[LangChain Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)、[LangGraph Subgraph persistence](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)。
