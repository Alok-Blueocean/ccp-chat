# LangGraph Concepts

General LangGraph knowledge — not used in this repo (which uses a fixed, non-looping `RetrievalPipeline`, see [01-retrieval-and-rag.md](01-retrieval-and-rag.md)), but a near-certain topic in any "how would you build an agent" interview question.

## Why LangGraph Exists

**What it is**
Chains built with LCEL (LangChain Expression Language) are directed acyclic graphs (DAGs) — data flows strictly forward from step A to step B to step C, with no way to loop back. Real agentic behavior, though, is inherently cyclical: call a tool, look at the result, decide whether to call another tool, maybe revise a plan, maybe loop back to an earlier step, and only then produce a final answer. Forcing that onto a DAG means burying the looping logic inside an opaque runtime (like the classic `AgentExecutor`), where you can't easily see or control what's happening between iterations. LangGraph's core move is to stop pretending an agent is a pipeline and instead model it as what it actually is: a **state machine** — nodes are states, edges (including edges that point backward) are transitions, and a shared state object is threaded through the whole thing. Because the graph is an explicit data structure, you can inspect it, test individual nodes, persist it mid-execution, and interrupt it — none of which is practical with a hidden loop.

**How it works**
- LCEL chains compose `Runnable` objects with `|`, producing a fixed forward-only pipeline; there is no way to conditionally jump backward.
- LangGraph represents control flow as a graph (`StateGraph`) with cycles allowed, so "call tool → observe → decide → maybe call another tool" is literally a loop in the graph rather than a `while` loop hidden inside library code.
- The graph is compiled into a runnable, but the structure (nodes, edges, routing functions) remains a first-class object you authored and can reason about.
- Because control flow is explicit, cross-cutting concerns — persistence, human approval, streaming, retries — attach to the graph itself instead of requiring framework-specific callback hooks.

**Example**
```python
# LCEL: cannot loop back — output of step N always feeds step N+1
chain = prompt | llm | output_parser  # strictly linear

# LangGraph: the same "call model, maybe call tool, maybe loop" is a real graph
from langgraph.graph import StateGraph, END

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", call_tool)
builder.add_edge("tools", "agent")  # <-- this backward edge is impossible in LCEL
builder.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
builder.set_entry_point("agent")
graph = builder.compile()
```

**Interview angle**
Q: Why not just keep using LCEL chains or `AgentExecutor` for agents?
A: LCEL is a DAG — it has no mechanism for cycles, and agent control flow (tool call → observe → decide → repeat) is inherently cyclical. `AgentExecutor` does support looping, but the loop is internal to the library, so you can't easily inspect, pause, checkpoint, or modify what happens between iterations. LangGraph makes the state machine explicit as a graph you author, so every one of those capabilities becomes possible instead of requiring you to fight the framework.

## StateGraph

**What it is**
`StateGraph` is the graph builder you construct an agent (or any multi-step workflow) from. You give it a schema for the shared **state** — usually a `TypedDict`, a `dataclass`, or a Pydantic model — that defines every field nodes can read or write. Critically, nodes never call each other directly or pass return values to one another the way functions do in normal code; instead, every node receives the *entire current state*, computes some update, and returns a partial dict that gets merged back into that same shared state before the next node runs. This indirection through one shared object is exactly what makes conditional branching and looping tractable: any node, or any routing function, can inspect the full history and decide what should happen next, because everything lives in one place.

**How it works**
- Define a state schema, typically `class AgentState(TypedDict): messages: Annotated[list, add_messages]`.
- Instantiate `builder = StateGraph(AgentState)`.
- Register nodes with `builder.add_node(name, fn)` where `fn(state) -> dict` returns a *partial* update.
- Register edges (`add_edge`, `add_conditional_edges`) to define transitions between nodes.
- Set `set_entry_point(name)` (or use the `START` constant) to mark where execution begins.
- Call `builder.compile()` to produce a runnable `CompiledGraph` you invoke with `.invoke(...)`, `.stream(...)`, etc.

**Example**
```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retries: int

def greet(state: AgentState) -> dict:
    return {"messages": [{"role": "assistant", "content": "hello"}]}

builder = StateGraph(AgentState)
builder.add_node("greet", greet)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)

graph = builder.compile()
result = graph.invoke({"messages": [], "retries": 0})
```

**Interview angle**
Q: How do nodes in a LangGraph pass data to each other if they don't call each other directly?
A: They don't communicate directly at all — every node reads from and writes partial updates to one shared state object defined by a schema (e.g. a `TypedDict`). The graph runtime merges each node's returned dict into that shared state before routing to the next node, so "communication" is really just "everyone agrees on one state shape."

## Nodes

**What it is**
A node is the basic unit of work in a LangGraph — a plain Python function (or a `Runnable`) with the signature `fn(state) -> dict`. It receives the current state and returns only the fields it wants to update, not the whole state. A node can do absolutely anything: call an LLM, call a tool, run a retrieval step, hit a database, do pure Python logic. There is no special "agent" magic baked into a node — it's an ordinary function, which is exactly why LangGraph graphs are easy to unit test in isolation (you can call a node function directly with a hand-built state dict and assert on its output, no graph required).

**How it works**
- Signature: `def node_fn(state: StateType) -> dict[str, Any]`, returning a partial update, not the full state.
- Nodes are registered via `builder.add_node("name", node_fn)`.
- A node can also be async (`async def`), enabling concurrent I/O-bound work when multiple nodes run in parallel branches.
- Because a node is "just a function," it composes with any existing code — an existing LCEL chain, a class method, a call to an external API — with no adapter layer required.
- Nodes can raise exceptions, which propagate up through `.invoke()`/`.stream()` unless you wrap them with your own retry logic or LangGraph's node-level retry policies.

**Example**
```python
from langchain_core.messages import AIMessage

def call_model(state: AgentState) -> dict:
    """A node that calls an LLM and appends the response to messages."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # partial update — merged via a reducer, not overwritten

def call_tool(state: AgentState) -> dict:
    """A node that executes whatever tool call the model just requested."""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
    return {"messages": [AIMessage(content=str(result))]}

# Unit test a node with zero graph machinery:
fake_state = {"messages": [{"role": "user", "content": "hi"}]}
assert "messages" in call_model(fake_state)
```

**Interview angle**
Q: What makes LangGraph nodes easy to test compared to logic buried inside an agent framework's internal loop?
A: A node is just a function that takes a state dict and returns a partial update — there's no framework object, no mocking a runtime, no spinning up the whole graph. You can call `node_fn(fake_state)` directly in a unit test and assert on the returned dict, the same as testing any pure function.

## Edges — Normal and Conditional

**What it is**
Edges define how control flow moves between nodes. A **normal edge** (`add_edge("A", "B")`) is unconditional — after node A finishes, node B always runs next. A **conditional edge** (`add_conditional_edges("A", routing_fn, mapping)`) is where the real power comes in: after node A finishes, a routing function inspects the *current state* and returns a label, which is looked up in a mapping to decide which node runs next. Crucially, a conditional edge can route back to a node earlier in the graph, which is precisely the mechanism that turns the graph into a loop — "call tool → back to agent → call tool again → back to agent → ... → done" is nothing more than a conditional edge pointing backward until the routing function decides to route to `END` instead.

**How it works**
- `add_edge(start, end)`: unconditional, always-taken transition.
- `add_conditional_edges(start, routing_fn, path_map)`: `routing_fn(state)` returns a key; `path_map` (a dict, or omitted if `routing_fn` returns node names directly) maps that key to a destination node name or `END`.
- The routing function receives the same state every node receives — it decides purely from state, with no side effects of its own (side effects belong in nodes).
- Loops are just conditional edges whose mapping includes a path back to an earlier node; there's no separate "loop" primitive.
- Newer LangGraph versions also support returning a `Command(goto=...)` from inside a node itself, letting a node choose its own next step without a separate routing function.

**Example**
```python
from langgraph.graph import END

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "continue"   # the model asked for a tool → loop back through "tools"
    return "end"             # the model gave a final answer → stop

builder.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)
builder.add_edge("tools", "agent")  # after running a tool, always go back to the agent
```

**Interview angle**
Q: How does LangGraph implement an agent's tool-calling loop — "keep calling tools until the model is done"?
A: A conditional edge out of the "agent" node inspects the latest message: if it contains tool calls, route to a "tools" node; otherwise route to `END`. A plain edge routes from "tools" back to "agent" after every tool execution. The loop isn't a special construct — it's just a conditional edge whose mapping happens to point back to an earlier node until the routing function decides otherwise.

## Reducers / State Merging

**What it is**
When a node returns a partial state update, the default behavior for each field is to **overwrite** whatever was there before with the new value. That's fine for scalar fields, but it's wrong for something like chat history: if every node that appends a message returned `{"messages": [new_msg]}` and the default overwrite behavior applied, each new message would *replace* the entire conversation instead of extending it. A **reducer** is a merge function attached to a specific state field (via `Annotated[type, reducer_fn]`) that tells LangGraph *how* to combine the old value and the new partial update for that field — for lists of messages, the built-in `add_messages` reducer appends (and also handles deduplication/updates by message ID). This is one of the most common practical "gotchas" for anyone who's actually built with LangGraph, which is exactly why it's worth having a concrete answer ready.

**How it works**
- Declared in the state schema via `Annotated[SomeType, reducer_function]`, e.g. `messages: Annotated[list[BaseMessage], add_messages]`.
- Without an annotation, LangGraph's default reducer is a plain overwrite: `new_value` replaces `old_value` entirely.
- `add_messages` (from `langgraph.graph.message`) is the standard reducer for chat history: it appends new messages, and if a new message shares an `id` with an existing one, it replaces that message in place (supporting edits/streaming updates) rather than duplicating it.
- You can write custom reducers for any merge semantics — e.g. `operator.add` for numeric accumulation, a custom function that merges two dicts, or one that de-duplicates a set of retrieved doc IDs.
- Reducers only apply to the *field* they're attached to — every field can have independent merge semantics in the same state object.

**Example**
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator

class AgentState(TypedDict):
    # Reducer: new messages are appended, not overwritten
    messages: Annotated[list, add_messages]
    # Reducer: numeric field accumulates instead of overwriting
    total_tokens_used: Annotated[int, operator.add]
    # No reducer: plain overwrite — last write wins
    current_step: str

def call_model(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],          # appended via add_messages
        "total_tokens_used": response.usage_metadata["total_tokens"],  # summed via operator.add
        "current_step": "model_called",  # overwritten
    }
```

**Interview angle**
Q: What's a subtle bug you'd hit if you forgot to use a reducer on a `messages` field?
A: Every node that returns `{"messages": [new_msg]}` would silently *replace* the whole conversation history with a single-message list instead of appending to it, because the default merge behavior is overwrite. Attaching `Annotated[list, add_messages]` to the field fixes it by telling LangGraph to append (and de-dupe by message ID) instead of overwrite.

## Checkpointing / Persistence

**What it is**
A checkpointer is a pluggable storage backend (in-memory `MemorySaver` for dev/testing, or durable options like `SqliteSaver` / `PostgresSaver` for production) that LangGraph uses to snapshot the *entire graph state* after every super-step, keyed by a `thread_id`. This is what gives an agent both durable multi-turn memory (the next call with the same `thread_id` resumes from exactly where the state left off — no need to manually re-pass history) and crash resilience (if the process dies mid-run, the last checkpoint is still on disk, and execution can resume from there rather than starting over). It's the same underlying idea as this repo's Postgres-backed `chat_memory.py`, but generalized to the framework level: it persists the *entire* graph state, not just a list of chat messages, and it does so automatically at every node boundary rather than via explicit save calls.

**How it works**
- Pass a checkpointer instance into `.compile(checkpointer=...)`.
- Every invocation must pass a `config={"configurable": {"thread_id": "..."}}` — the checkpointer uses this to know which conversation/session's state to load and save.
- After each node finishes (each "super-step"), the checkpointer writes a snapshot of the full state, tagged with that thread ID and a monotonically increasing checkpoint id.
- `.get_state(config)` retrieves the latest checkpoint for a thread; `.get_state_history(config)` retrieves every checkpoint, enabling "replay from an earlier point" or "what did the state look like before step 3" debugging.
- Because interrupts (below) rely on the graph being resumable exactly where it paused, checkpointing is a hard prerequisite for human-in-the-loop workflows, not just a convenience.

**Example**
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # swap for SqliteSaver / PostgresSaver in production
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-42-session-1"}}

# First turn — starts fresh, checkpointer saves state after it finishes
graph.invoke({"messages": [{"role": "user", "content": "Hi, I'm Alok"}]}, config)

# Second turn, same thread_id — automatically resumes from the saved state,
# so the model still knows the user's name without you re-sending history
graph.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
```

**Interview angle**
Q: How does LangGraph give an agent memory across turns, and how is that different from just re-sending chat history yourself?
A: It's checkpointing, not a special "memory" object — a checkpointer snapshots the entire graph state after every step, keyed by `thread_id`, and automatically reloads it on the next invocation with the same thread ID. Unlike manually re-sending history, this persists *all* state fields (not just messages) and supports resuming mid-execution after a crash or an interrupt, which manual history replay can't do.

## Human-in-the-Loop (Interrupts)

**What it is**
LangGraph lets you pause execution at a specific point in the graph and wait for external — typically human — input before continuing, using the `interrupt()` function inside a node. This is the standard mechanism for "require approval before this agent does something irreversible," e.g. sending an email, executing a database write, or making a purchase. Because interrupts rely on checkpointing, the paused state is durable: a human can approve the action seconds, minutes, or literally days later, and when the graph is resumed (via `Command(resume=...)`), it picks up exactly where it left off rather than needing to replay from the start.

**How it works**
- Call `interrupt(payload)` inside a node; this raises a special exception that halts graph execution at that exact point and surfaces `payload` to the caller.
- The graph must be compiled with a checkpointer — the interrupt relies on the last checkpoint to know how to resume.
- The caller inspects the paused state via `graph.get_state(config)` (its `.next` attribute shows which node is waiting) and decides what to do.
- To resume, invoke the graph again with `Command(resume=<value>)` instead of a fresh input — the `interrupt()` call effectively "returns" that resume value inside the node, and execution continues from there.
- This pattern is commonly combined with `interrupt_before=[...]` / `interrupt_after=[...]` compile-time options for static (always-pause-here) interrupts, versus the dynamic `interrupt()` call for conditional pauses decided inside node logic.

**Example**
```python
from langgraph.types import interrupt, Command

def request_approval(state: AgentState) -> dict:
    action = state["proposed_action"]
    decision = interrupt({"question": f"Approve this action? {action}"})
    if decision == "approve":
        return {"status": "approved"}
    return {"status": "rejected"}

# --- Caller side ---
config = {"configurable": {"thread_id": "order-99"}}
graph.invoke({"proposed_action": "refund $500"}, config)
# Execution is now paused inside request_approval; inspect it:
state = graph.get_state(config)
print(state.next)  # ('request_approval',) — this node is waiting

# A human reviews and approves, then execution resumes exactly where it paused:
graph.invoke(Command(resume="approve"), config)
```

**Interview angle**
Q: How would you make an agent pause and wait for a human to approve an irreversible action, like a real refund or a production deploy?
A: Call `interrupt(...)` inside the node right before the risky action — this halts the graph and persists its state via the checkpointer. A human reviews the pending action out-of-band (e.g. in a UI), and when they approve, you resume with `graph.invoke(Command(resume=decision), config)`, which continues execution exactly where it paused, no replay needed.

## Streaming

**What it is**
Instead of blocking until an entire multi-step graph run finishes, LangGraph can stream results as execution progresses, at several different granularities: per-node state updates (`stream_mode="updates"`), full state snapshots after each step (`"values"`), or token-level output from LLM calls happening inside nodes (`"messages"`). This matters a lot for UX on any multi-step agent — a user watching a spinner for 15 seconds while five tool calls happen sequentially feels broken; a user watching "searching docs... calling calculator... drafting answer..." feels responsive, even though the total latency is identical.

**How it works**
- `graph.stream(input, config, stream_mode="updates")` yields `{node_name: partial_state_update}` after each node finishes — good for showing high-level progress ("now running: retrieval").
- `stream_mode="values"` yields the full accumulated state after each step, useful if a consumer wants the whole picture each time rather than a diff.
- `stream_mode="messages"` streams token-by-token LLM output from *inside* any node that calls a chat model, giving true token-level streaming even though the node itself is one step of a larger graph.
- Multiple modes can be requested together, e.g. `stream_mode=["updates", "messages"]`, letting a UI show both step progress and live token output simultaneously.
- Async equivalents (`astream`) exist for use inside async web frameworks (FastAPI, etc.), which matters when a graph is serving concurrent requests.

**Example**
```python
config = {"configurable": {"thread_id": "session-7"}}

# See each node's output as it completes
for update in graph.stream({"messages": [user_msg]}, config, stream_mode="updates"):
    for node_name, partial_state in update.items():
        print(f"[{node_name}] produced: {partial_state}")

# See token-level output from whichever node is currently calling the LLM
for msg_chunk, metadata in graph.stream(
    {"messages": [user_msg]}, config, stream_mode="messages"
):
    print(msg_chunk.content, end="", flush=True)
```

**Interview angle**
Q: A multi-step agent takes 15 seconds to run five tool calls. How do you keep the UI from feeling frozen?
A: Stream instead of blocking on `.invoke()`. `stream_mode="updates"` surfaces each node's output as it completes so the UI can show step-by-step progress ("searching... calculating... drafting..."), and `stream_mode="messages"` gives token-level streaming of whatever LLM call is currently running inside a node — you can combine both so the user sees real-time progress and real-time text generation at once.

## Multi-Agent Patterns in LangGraph

**What it is**
LangGraph doesn't have a distinct "multi-agent" feature — multi-agent systems fall out naturally from the fact that a compiled graph is itself just a `Runnable`, so it can be invoked as a single node inside a *parent* graph (a "subgraph"). The two dominant patterns built on this: the **supervisor pattern**, where one router/supervisor node examines the task and dispatches it to the right specialist sub-agent (each potentially its own subgraph), collecting and synthesizing their outputs; and **hierarchical teams**, where supervisors themselves report to a higher-level supervisor, forming a tree of graphs of graphs. Because composition is native, there's no ceiling on how deep this nesting can go — a "team" node in a top-level graph can itself be a supervisor graph managing three specialist subgraphs.

**How it works**
- Any compiled graph (`builder.compile()`) is a `Runnable` and can be added as a node in another graph: `parent_builder.add_node("research_team", research_subgraph)`.
- Supervisor pattern: a router node uses an LLM call (or simple logic) to decide which specialist to invoke next, via a conditional edge, then loops back to the supervisor after each specialist responds — structurally identical to the tool-calling loop, just with "specialist agents" standing in for "tools."
- State needs to be either shared (same schema across parent and subgraph) or explicitly transformed at the subgraph boundary if the subgraph has its own distinct state schema.
- Hierarchical teams are just supervisor graphs nested inside other supervisor graphs — no special primitive, just recursion of the same pattern.
- Common alternative: a fully decentralized "swarm" pattern where agents hand off directly to each other (via `Command(goto=...)`) rather than routing through a central supervisor.

**Example**
```python
# Two specialist subgraphs, each a complete compiled StateGraph
research_agent = research_builder.compile()
writer_agent = writer_builder.compile()

def supervisor(state: TeamState) -> str:
    """Router: decide which specialist should act next, or finish."""
    if not state.get("research_done"):
        return "researcher"
    if not state.get("draft_written"):
        return "writer"
    return "END"

team_builder = StateGraph(TeamState)
team_builder.add_node("researcher", research_agent)  # a whole subgraph as one node
team_builder.add_node("writer", writer_agent)
team_builder.add_conditional_edges(
    "supervisor_router", supervisor, {"researcher": "researcher", "writer": "writer", "END": END}
)
team_builder.add_edge("researcher", "supervisor_router")
team_builder.add_edge("writer", "supervisor_router")
```

**Interview angle**
Q: How does LangGraph handle multi-agent orchestration — is there a special "multi-agent" API?
A: No — a compiled `StateGraph` is just a `Runnable`, so it can be used as a node inside a bigger graph. The supervisor pattern is a router node that dispatches to specialist subgraphs via a conditional edge and loops back after each one responds, structurally the same as a tool-calling loop. Hierarchical teams are just supervisor graphs nested inside other supervisor graphs — multi-agent composition is a consequence of "graphs can contain graphs," not a distinct feature.

## Tool-Calling Node and the ReAct Loop

**What it is**
The most common concrete LangGraph pattern — the one nearly every "build a tool-using agent" tutorial converges on — pairs an "agent" node (an LLM bound to a set of tools via `.bind_tools()`) with a "tools" node that actually executes whatever tool calls the model requested, wired together with the conditional-edge loop described above. LangGraph ships a prebuilt `ToolNode` that handles the mechanical part (reading `tool_calls` off the last AI message, invoking the matching tool by name, wrapping results as `ToolMessage`s) so you don't hand-write that dispatch logic yourself. This is the ReAct ("Reason + Act") loop made explicit: the model reasons about what to do, optionally acts via a tool, observes the result, and reasons again — repeating until it produces a final answer with no further tool calls.

**How it works**
- Bind tools to the model once: `model_with_tools = llm.bind_tools([search, calculator])`.
- The "agent" node just calls `model_with_tools.invoke(state["messages"])`, appending whatever it returns (possibly containing `tool_calls`) to state.
- `ToolNode(tools)` is a prebuilt node that takes the last message, executes every requested tool call, and returns the results as `ToolMessage` objects appended to `messages`.
- A conditional edge (often the prebuilt `tools_condition` helper) checks whether the last AI message has any `tool_calls`; if yes, route to `"tools"`, otherwise route to `END`.
- `langgraph.prebuilt.create_react_agent(model, tools, checkpointer=...)` packages this entire pattern (agent node + `ToolNode` + conditional loop + checkpointing) into a single call for the common case, which is worth knowing exists even if you'd hand-roll the graph for anything nontrivial.

**Example**
```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END

tools = [search_tool, calculator_tool]
model_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    return {"messages": [model_with_tools.invoke(state["messages"])]}

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))     # prebuilt dispatcher
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # routes to "tools" or END
builder.add_edge("tools", "agent")
react_graph = builder.compile()

# Equivalent one-liner for the common case:
# from langgraph.prebuilt import create_react_agent
# react_graph = create_react_agent(llm, tools, checkpointer=checkpointer)
```

**Interview angle**
Q: Do you have to hand-write the logic that executes whichever tool the model asked for?
A: No — LangGraph ships `ToolNode`, which reads the `tool_calls` off the last AI message, invokes the matching tool by name with the given arguments, and returns the results as `ToolMessage`s, and `tools_condition`, a ready-made routing function for the "does the last message have tool calls" check. For the common single-agent ReAct case there's also `create_react_agent`, which assembles the whole agent-node/ToolNode/conditional-loop graph for you in one call.

## LangGraph vs. Classic `AgentExecutor`

**What it is**
`AgentExecutor` (LangChain's older agent runtime) is a fixed, mostly opaque ReAct loop: you give it a model, a set of tools, and a prompt, and it internally loops "think → act → observe" until the model stops requesting tools, exposing only a handful of hooks (callbacks, max iterations, early-stopping method) to influence that loop from outside. LangGraph exposes the *same* underlying loop as an explicit set of nodes and edges you author yourself, meaning every step — including ones `AgentExecutor` doesn't expose at all, like "pause here for approval" or "checkpoint state after this step" — becomes something you can inspect, test, and modify individually. The tradeoff is real, though: LangGraph requires you to actually design a state machine (state schema, nodes, routing logic) instead of getting a working agent for free from three constructor arguments.

**How it works**
- `AgentExecutor`: `agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=...)` — the loop, error handling, and iteration limits are all internal to the class.
- LangGraph: the loop is external and visible — you decide exactly what "one iteration" means, what state persists across iterations, and what happens at each edge.
- `AgentExecutor` has no native checkpointing, no native interrupt/human-approval mechanism, and no native support for arbitrary branching (e.g., "if the tool result looks wrong, revise the plan instead of proceeding") — anything beyond the built-in loop shape requires subclassing or workarounds.
- LangGraph graphs are unit-testable node by node; `AgentExecutor`'s loop is essentially a black box you can only test end-to-end.
- LangChain has in fact deprecated `AgentExecutor` in favor of LangGraph for new agent development, which is itself a useful fact to cite.

**Example**
```python
# Classic AgentExecutor — a few lines, but the loop is a black box
from langchain.agents import AgentExecutor, create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5)
result = executor.invoke({"input": "What's 12% tip on $84?"})

# LangGraph — more setup, but every step is visible/testable/pausable
# (using the react_graph built in the previous section)
result = react_graph.invoke(
    {"messages": [{"role": "user", "content": "What's 12% tip on $84?"}]},
    config={"configurable": {"thread_id": "t1"}},
)
```

**Interview angle**
Q: Should every agent be built in LangGraph, or is `AgentExecutor` (or an equivalent simple loop) ever the right call?
A: Not every agent needs LangGraph's overhead — if it's a simple single-tool-call agent with no need for persistence, interruption, or custom branching, a simpler loop or `AgentExecutor`-style abstraction is less upfront design work. LangGraph earns its complexity when you need things `AgentExecutor` can't give you at all: durable checkpointing, human-in-the-loop pauses, non-trivial branching/looping logic, or multi-agent composition. Worth noting LangChain itself has been steering new agent development toward LangGraph, so it's the more future-proof default for anything beyond the trivial case.
