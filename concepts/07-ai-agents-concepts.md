# AI Agent Concepts

General agent theory and patterns — this repo's retrieval pipeline is *not* an agent (it's a fixed pipeline, no LLM-driven control flow), which is itself a useful distinction to be able to draw.

## What Makes Something an "Agent"

**What it is**

An agent is an LLM that decides its *own next action* inside a loop — which tool to call, with what arguments, and when to stop — rather than executing a hardcoded sequence of steps that a human engineer wrote in advance. The defining feature is *where control flow lives*. In a normal program (including most "AI-powered" pipelines), the order of operations is written in your code: step 1 always runs, then step 2, then step 3. In an agent, the model's output on each turn determines what happens next — the "program" is effectively generated one step at a time by the LLM itself. This is a spectrum, not a binary: a single LLM call with no tools is clearly not an agent, and a system that can call any of ten tools in any order, any number of times, deciding for itself when it has enough information to answer, clearly is. Most real systems sit somewhere in between (e.g. a fixed pipeline with one LLM-driven branch point), so the useful skill is being able to point at the exact place where control passes from code to model.

**How it works**

- Ask: "who decides what happens next — a line of code, or a token the model generated?" If it's the model, that segment is agentic.
- A single LLM call that returns an answer is not an agent — there's no loop, no decision about "what to do next."
- A pipeline that always does retrieve → generate → return is not an agent, even though step 2 is an LLM call, because the *sequence* is fixed by code, not chosen by the model.
- An agent typically needs: (1) tools it can choose to invoke, (2) a loop that keeps running until some condition is met, (3) the model's own output steering which branch of the loop is taken.
- Degree of "agentic-ness" scales with how much of the control flow is delegated: one decision point (e.g. "should I retrieve or answer directly?") is mildly agentic; a fully open-ended ReAct loop with a dozen tools is highly agentic.

**Example**

This repo's chatbot: user asks a question → code always calls the retriever → code always stuffs the top-k chunks into a prompt → code always calls the LLM once → code returns the answer. Every arrow in that chain is drawn by the engineer, not the model — swap it for a different fixed template and it's still the same shape of system. Contrast with a support-ticket agent: it receives the same question, but the *model* decides "I should first look up the customer's account tier, then check the refund-policy tool, then decide whether to escalate to a human" — and it could just as easily have decided to answer directly, or to call a third tool first. That branching decision, made by the model at runtime rather than by the engineer at write-time, is what makes the second system an agent and the first one not.

**Interview angle**

Q: "Is your RAG chatbot an agent?"
A: No — it's a fixed retrieval pipeline. Retrieve, then generate, then return, in that exact order, every time, decided by code. An agent would mean the model itself decides whether to retrieve, how many times, using which of several tools, and when it has enough to answer. The distinction matters because "agent" implies a specific reliability/cost/latency tradeoff (more flexible, but less predictable and more expensive) that a fixed pipeline deliberately avoids.

---

## ReAct (Reason + Act)

**What it is**

ReAct ("Reasoning and Acting," Yao et al. 2022) is a prompting pattern where the model alternates between writing out an explicit reasoning step ("Thought: ...") and taking an action ("Action: tool_name(args)"), then reads the result of that action ("Observation: ...") before reasoning about what to do next. Instead of asking the model to produce one plan up front and execute it blindly, ReAct interleaves thinking and doing turn by turn, so the model can react to surprises — a tool returning something unexpected, an error, a value that changes the plan — instead of committing to a brittle upfront plan. It was one of the first patterns to show that giving a model a visible "scratchpad" to reason in, combined with the ability to take real actions and see their results, substantially improves multi-step task performance over either reasoning-only (chain-of-thought) or acting-only approaches. Almost every modern tool-using agent framework's core loop is a ReAct loop under the hood, even when the "Thought/Action/Observation" text format has been replaced by structured tool-call objects.

**How it works**

- The model is prompted (or trained) to alternate: `Thought → Action → Observation → Thought → Action → Observation → ... → Final Answer`.
- **Thought**: free-text reasoning about what's known so far and what to do next — not shown to the end user, but critical for the model's own accuracy.
- **Action**: a call to a specific tool with specific arguments, expressed either as text (`Action: search("...")`) in the original paper or as a structured tool call in modern implementations.
- **Observation**: the raw result of executing that action, appended back into the context for the next Thought.
- The loop terminates when the model emits a designated "final answer" action (or a stop condition like max iterations is hit).
- Because reasoning happens *between* actions rather than only before the first one, the model can course-correct: notice a tool failed, notice a value implies a different next step, or decide it already has enough information to stop early.

**Example**

Task: "What's the weather-adjusted price of a large latte in Austin right now?" (fictional business rule: price gets a $0.50 surcharge if it's raining.)

```
Thought: I need today's weather in Austin, then the base price of a large
latte, then apply the rain surcharge rule if it's raining.
Action: get_weather(city="Austin")
Observation: {"city": "Austin", "condition": "rain", "temp_f": 68}

Thought: It's raining in Austin, so the $0.50 surcharge applies. Now I
need the base price of a large latte.
Action: get_price(item="latte", size="large")
Observation: {"item": "latte", "size": "large", "base_price": 4.75}

Thought: Base price is $4.75, condition is rain, so weather-adjusted
price = 4.75 + 0.50 = 5.25. I have everything I need to answer.
Action: final_answer("$5.25 (base $4.75 + $0.50 rain surcharge)")
```

Three loop iterations: two tool calls, each followed by reasoning that used the observation to decide the next step, then a final answer. Note the second Thought explicitly used the *result* of the first Observation (rain → surcharge applies) — that dependency is exactly what a fixed, non-looping pipeline can't express unless it's hardcoded in advance.

**Interview angle**

Q: "Why interleave reasoning and acting instead of having the model plan all its tool calls up front and then execute them?"
A: Because upfront plans can't account for what a tool actually returns. If step 2's plan depends on step 1's result (e.g. "surcharge only applies if it's raining"), a model that commits to a full plan before seeing any observations has to guess. ReAct lets the model re-reason after every observation, trading a few extra tokens/latency for the ability to adapt mid-task — which is usually worth it unless the task is simple enough that the plan truly can't change.

---

## Tool / Function Calling

**What it is**

Tool calling (also called function calling) is the mechanism by which a model is given a set of typed tool definitions — name, natural-language description, and a JSON-schema for its arguments — and, instead of only producing free-form text, can produce a structured, machine-parseable request to invoke one of those tools with specific argument values. The model itself never executes the tool; it only emits the *intent* to call it (name + arguments) as structured output, and the calling application is responsible for actually running the function, then feeding the result back into the conversation as a new message. This is fundamentally a constrained-decoding / structured-output problem — the model is trained or prompted to produce output conforming to a schema — not a separate capability bolted onto the model. The tool's *description* is itself part of the prompt: a vague or ambiguous description causes the same kind of misuse a vague piece of human-facing documentation would (wrong tool chosen, wrong arguments guessed), so writing good tool descriptions is a real engineering task, not an afterthought.

**How it works**

- Each tool is declared with a schema: name, description, and a JSON Schema for parameters (types, required fields, enums, descriptions per field).
- The full set of tool schemas is sent to the model alongside the conversation on every turn where tool use should be considered.
- The model outputs either normal text, or a structured tool-call object (name + arguments matching the schema) — many APIs support emitting multiple tool calls in one turn for parallelizable actions.
- The calling code validates the arguments (ideally against the same schema), executes the real function/API/DB call, and returns the result as a "tool result" message.
- That tool result is appended to the conversation history, and the model is invoked again — this is what turns a single tool call into a loop (see "The Agentic Loop" below).
- Good tool descriptions specify not just *what* the tool does but *when* to use it and *what it doesn't do*, since the model relies entirely on the description (not the implementation) to decide when to call it.

**Example**

Tool schema given to the model:

```json
{
  "name": "get_weather",
  "description": "Get the current weather conditions for a city. Use this when the user's question depends on today's weather. Does not provide forecasts for future dates.",
  "input_schema": {
    "type": "object",
    "properties": {
      "city": { "type": "string", "description": "City name, e.g. 'Austin'" },
      "units": { "type": "string", "enum": ["fahrenheit", "celsius"], "default": "fahrenheit" }
    },
    "required": ["city"]
  }
}
```

Model's structured tool-call output (this is what the API returns, not text the model "typed"):

```json
{
  "type": "tool_use",
  "id": "toolu_01Ab23Cd",
  "name": "get_weather",
  "input": { "city": "Austin", "units": "fahrenheit" }
}
```

The application executes `get_weather(city="Austin", units="fahrenheit")`, gets back `{"condition": "rain", "temp_f": 68}`, and sends that back as a `tool_result` message referencing `toolu_01Ab23Cd` so the model knows which call it answers.

**Interview angle**

Q: "How is tool calling actually implemented under the hood — is it a special model feature?"
A: It's constrained/structured output, not a bolted-on capability — the model is trained to emit JSON matching a provided schema instead of (or alongside) prose, and the API layer parses that into a typed object. The model never runs the tool; your application code does, then feeds the result back in as another message. This is also why tool descriptions matter so much: they're the only signal the model has for *when* and *how* to use a tool, so a vague description causes misuse the same way vague documentation would for a human.

---

## The Agentic Loop

**What it is**

The agentic loop is the control structure that turns a single tool call into an autonomous, multi-step process: perceive the current state (read the conversation and the latest tool result), plan (decide the next action), act (call a tool), observe (read what came back), and repeat until some stop condition is met. It's the runtime implementation of the ReAct pattern — ReAct describes the reasoning *style* inside each turn, the agentic loop is the *engineering scaffold* (the actual `while` loop in code) that keeps invoking the model, executing whatever tool it asks for, and feeding results back, turn after turn. The single most important engineering detail is the stop condition: without one, a model that gets confused, stuck, or subtly wrong can call tools indefinitely, burning cost and latency with no user-visible progress. Every production agent framework's core value-add is largely *this loop plus its guardrails* — max iterations, budgets, and a well-defined "done" signal — rather than anything exotic in the model itself.

**How it works**

- **Perceive**: gather the current context — conversation history, most recent tool result, any system state.
- **Plan**: send everything to the model; it decides the next action (call a tool, or emit a final answer).
- **Act**: if a tool was requested, the application executes it (with argument validation, timeouts, and error handling).
- **Observe**: the tool's result (or error) is appended to the context as a new message.
- **Repeat**, feeding the growing context back to the model each iteration.
- **Stop conditions** — at least one of: the model emits a designated "final answer" / "done" action; a max-iteration count is reached; a cost or token budget is exhausted; a wall-clock timeout fires; a human-in-the-loop gate blocks further action pending approval.

**Example** — pseudocode of the loop with an explicit stop condition:

```python
def run_agent(user_message, tools, max_iterations=8, max_cost_usd=0.50):
    messages = [{"role": "user", "content": user_message}]
    total_cost = 0.0

    for iteration in range(max_iterations):
        response = llm.call(messages, tools=tools)
        total_cost += response.cost
        messages.append(response.message)

        if response.stop_reason == "final_answer":
            return response.final_answer          # normal exit

        if total_cost > max_cost_usd:
            return "Stopped: cost budget exceeded"  # budget exit

        # act: execute every tool call the model requested this turn
        for call in response.tool_calls:
            try:
                result = execute_tool(call.name, call.arguments)
            except Exception as e:
                result = {"error": str(e)}          # errors become observations too
            messages.append(tool_result_message(call.id, result))

    return "Stopped: max iterations reached"          # iteration-cap exit
```

Three distinct stop conditions are visible: a clean "final_answer" exit, a cost-budget exit, and an iteration-cap exit as the last-resort backstop. Note that a failed tool call doesn't crash the loop — the error itself becomes an observation the model can reason about on the next turn (e.g. "that tool failed, let me try different arguments").

**Interview angle**

Q: "How do you prevent an agent from running forever or spiraling in cost?"
A: Layer multiple stop conditions rather than relying on one: a hard max-iteration cap as the ultimate backstop, a cost or token budget checked every turn, and ideally a "final answer" tool the model is instructed to call as soon as it has enough information, so the happy path exits early rather than always burning the full budget. In production you'd also add per-user rate limits and alerting on agents that repeatedly hit the iteration cap, since that's a strong signal something's wrong with the prompt or tools, not just an unlucky run.

---

## Planning Strategies

**What it is**

Planning strategy refers to *how much* an agent commits to a course of action before executing versus how often it re-evaluates. At one end, single-shot planning has the model produce one complete plan up front, then execute every step blindly without reconsidering; at the other, plan-and-execute (or "replanning") has the model re-plan after every single step using the latest observations; in between and beyond, tree/graph search strategies like Tree-of-Thought have the model explore multiple candidate next-steps or full solution branches, evaluate them, and prune the weak ones before committing. This is fundamentally a cost/adaptability tradeoff, not a "which one is objectively better" question: replanning after every step is more robust to surprises but means an LLM call per step even when nothing unexpected happened; a single upfront plan is cheap and fast but brittle the moment reality deviates from what the model assumed when it planned. Tree search strategies buy even more robustness (exploring several possible paths instead of committing to one) at a multiplicative cost in LLM calls, so they're reserved for tasks where a wrong first guess is expensive and getting it right matters more than speed or cost.

**How it works**

- **Single-shot plan-then-execute**: model produces a full ordered list of steps once; code executes them in order without asking the model to reconsider. Cheapest, fastest, most brittle.
- **Plan-and-execute (replanning)**: model produces a plan, executes step 1, feeds the result back, and is asked "given this, what's the plan now?" — potentially revising all remaining steps. This is essentially ReAct at a coarser granularity (a full plan, not a single tool call, is reconsidered each round).
- **Tree/graph search (e.g. Tree-of-Thought)**: at each decision point the model generates *multiple* candidate next steps (or reasoning branches), a scoring step (model self-evaluation, a heuristic, or a separate verifier) ranks them, weak branches are pruned, and the search continues down the strongest branch(es) — optionally backtracking if a branch dead-ends.
- Choice of strategy is driven by: how likely is the plan to need revision mid-execution (uncertain environment → replan more), and how costly is a wrong step (expensive/irreversible actions → search more, or add a human checkpoint).

**Example**

Task: "Plan and book a 3-day trip to Austin within $800."

- *Single-shot*: model outputs "1) book flight, 2) book hotel, 3) book 2 restaurant reservations" and the code executes all three, in order, no matter what the flight ends up costing.
- *Plan-and-execute*: model plans the same 3 steps, but after step 1 returns "flight was $450, more than expected," it's asked to replan — it now revises step 2 to a cheaper hotel to stay within budget, instead of blindly booking the originally planned one.
- *Tree search*: before booking anything, the model generates three candidate itineraries (budget/mid/splurge), a scoring pass estimates total cost and "trip quality" for each, the over-budget splurge branch is pruned, and the agent proceeds with the best remaining candidate.

**Interview angle**

Q: "Which planning strategy should an agent use?"
A: Frame it as a tradeoff, not a ranking: single-shot is cheap but brittle to surprises; plan-and-execute costs more LLM calls but adapts as new information arrives; tree/graph search costs the most (multiplicatively — you're generating and scoring several branches) but is worth it when a wrong early decision is expensive or hard to undo. The right choice depends on how uncertain the environment is and how costly a bad step is, not on which pattern is "more advanced."

---

## Memory Types

**What it is**

"Memory" in an agent context is a set of mechanisms for getting relevant information *back into the model's context window*, since the underlying LLM itself is stateless — it has no memory between calls, only whatever text is in the prompt it's given right now. Short-term memory is the current context window: the conversation so far, plus any tool results accumulated during this run. Long-term memory is external storage — a vector database, key-value store, or plain database row — that the agent can explicitly read from and write to, so information can survive across sessions or be shared across users/agents. Episodic memory is a specific kind of long-term memory: logs of entire past task *runs* (not just facts) that an agent can retrieve and learn from — e.g. "last time I tried this approach it failed for this reason, avoid it" — used for reflection and self-improvement rather than simple fact lookup. The critical distinction to keep straight in an interview is "the model remembers" versus "the system re-feeds relevant history into context" — the former is not actually true of any current LLM; the latter is what every memory system, however sophisticated, ultimately reduces to.

**How it works**

- **Short-term / working memory**: literally the token sequence in the current context window — conversation turns, tool call/results, scratchpad reasoning. Bounded by the model's context window size; grows every turn until pruned or summarized.
- **Long-term memory**: an external store (vector DB for semantic similarity search, Redis/KV store for structured facts, relational DB for structured records) that the agent queries via a tool call, gets results back as text, and those results are then injected into short-term memory for that turn. The store persists independent of any single conversation.
- **Episodic memory**: long-term storage specifically of past *episodes* (full task attempts, with outcomes) rather than isolated facts — retrieved to inform "have I tried something like this before, what happened?"
- Writing to long-term memory is itself often an agent action: the agent (or a separate summarization step) decides what's worth persisting and calls a "save_memory" style tool, rather than everything being saved automatically.
- Retrieval into context is usually similarity-search or recency-based, same as RAG retrieval — memory systems and RAG retrieval are architecturally the same idea (fetch relevant text, inject into prompt) applied to different sources (a knowledge base vs. an agent's own history).

**Example**

This repo's `chat_memory.py` is a concrete short-term memory implementation: it's not the model "remembering" anything — it's the application re-injecting the last N conversation turns into the prompt on every call, backed by Postgres for persistence across page reloads. That's short-term memory with a durable backing store, but it is *not* long-term memory in the agentic sense, because nothing about it is selectively retrieved by relevance or written to based on an agent's own judgment of what's worth keeping — it just replays recent turns verbatim. A long-term-memory agent, by contrast, might have a `save_fact(text)` tool it calls mid-conversation ("the user prefers metric units") and a `recall(query)` tool that does a vector search over everything it has ever saved, across all past sessions with that user — closer to genuine cross-session memory than a recent-turns buffer.

**Interview angle**

Q: "Does the model actually remember previous conversations?"
A: No — the model is stateless between calls; every apparent "memory" is the application re-feeding relevant text back into the prompt. Short-term memory is just the current context window; long-term memory means an external store (vector DB, KV store) that the agent explicitly reads from and writes to via tool calls, with retrieved results injected into context just like RAG retrieval. The mechanism is identical to RAG — fetch relevant text, put it in the prompt — the only difference is *what* you're retrieving from (a knowledge base versus the agent's own history).

---

## Single-Agent vs. Multi-Agent

**What it is**

A single-agent system has one LLM, with one set of tools and one context, doing the entire task itself, looping until done. A multi-agent system splits the work across multiple specialized agents — e.g. a researcher agent, a coder agent, and a reviewer agent — coordinated either by an orchestrator agent that delegates subtasks and collects results, or via direct hand-off where one agent passes control (and context) straight to the next. The appeal of multi-agent is the same as the appeal of splitting a large function into smaller ones in normal software: each individual agent gets a narrower, more focused prompt and tool set, which tends to make its behavior more reliable and easier to reason about than one agent juggling everything. The cost is coordination overhead and a new failure surface — agents can misunderstand the task handed to them, an orchestrator can misroute a subtask to the wrong specialist, and debugging requires tracing across multiple agents' contexts instead of one.

**How it works**

- **Orchestrator pattern**: a top-level agent decomposes the task, dispatches subtasks to specialist agents (as if they were just another kind of "tool"), collects their results, and synthesizes a final answer.
- **Hand-off pattern**: agent A decides it's done with its part and explicitly transfers the conversation (and relevant context) to agent B, which continues from there — no central coordinator.
- Specialist agents typically have a narrower tool set and a prompt scoped to their one job, which is exactly what makes each of them more reliable in isolation.
- Coordination overhead shows up as: extra LLM calls (the orchestrator itself burns tokens deciding who to delegate to), context-passing losses (a specialist doesn't automatically see everything the orchestrator saw, unless it's explicitly passed along), and misrouting (orchestrator delegates to the wrong specialist, or a specialist gets a subtask it doesn't have the tools to actually complete).
- Multi-agent is not free parallelism by default — agents whose subtasks are dependent still have to run sequentially; only independent subtasks can genuinely run concurrently.

**Example**

A code-review multi-agent setup: an *orchestrator* receives "review this PR," delegates "summarize what changed" to a *summarizer* agent, delegates "check for security issues" to a *security* agent (with a narrower tool set: static-analysis tools, CVE lookup, no code-editing tools at all), and delegates "check test coverage" to a *test* agent. Each specialist returns a focused report; the orchestrator synthesizes them into one review comment. Compare to a single-agent version: one agent with *all* those tools available at once, deciding for itself which checks to run in which order — simpler to build and debug, but that one agent's prompt has to cover every concern at once, which in practice tends to make it skip things a narrowly-scoped specialist wouldn't.

**Interview angle**

Q: "When would you *not* use a multi-agent architecture?"
A: When the task doesn't decompose cleanly into independent or clearly-sequential sub-roles — if every subtask needs the full context anyway, splitting it up just adds coordination overhead (extra LLM calls, context-passing losses, misrouting risk) without the reliability benefit of narrower per-agent prompts. Multi-agent earns its overhead when specialization measurably improves each part's reliability or lets independent parts run in parallel; if neither is true, one well-scoped single agent is simpler to build, debug, and reason about.

---

## Common Agent Failure Modes

**What it is**

Agents fail in a handful of characteristic ways that are worth naming precisely, because "agents can be unreliable" is not an answer an interviewer will find convincing — the specific failure mode and its specific mitigation is. The main ones: infinite or near-infinite tool-call loops (the agent keeps calling tools without converging on an answer); hallucinated tool arguments (the model calls a real tool with plausible-looking but fabricated argument values, e.g. inventing an order ID that doesn't exist); context window overflow from accumulating tool results (every observation gets appended to context, and a long-running agent eventually runs out of room, or relevant early information gets pushed out / drowned out); tool misuse (the right tool is called, but with the wrong arguments or in a situation it wasn't meant for); and silent partial failure (a step fails or returns incomplete data, but nothing in the trace flags it, so the agent confidently proceeds as if it succeeded, and the final answer looks fine but is wrong).

**How it works / mitigations**

- **Infinite loops** → mitigated by max-iteration caps and cost/token budgets (see "The Agentic Loop"), plus detecting repeated identical tool calls as an early-exit signal.
- **Hallucinated arguments** → mitigated by validating tool call arguments against the tool's JSON schema *and* against real referential data before executing (e.g. does this order ID actually exist?) rather than trusting the model's output blindly.
- **Context overflow** → mitigated by summarizing or pruning old tool results out of context once they're no longer needed, keeping only the final relevant extract rather than the full raw payload of every past observation.
- **Tool misuse** → mitigated by tightening tool descriptions (explicit "use this when / don't use this for"), and by validating pre-conditions before execution rather than after.
- **Silent partial failure** → mitigated by making tools raise explicit, structured errors instead of returning ambiguous partial data, and by having the agent (or a supervising check) explicitly verify success rather than assume it.

**Example — a broken trace, and how it's caught**

```
Thought: I need the customer's order status.
Action: get_order_status(order_id="ORD-88213")
Observation: {"error": "order not found"}

Thought: That didn't work, let me try a slightly different ID.
Action: get_order_status(order_id="ORD-88214")
Observation: {"error": "order not found"}

Thought: Still not found, let me try again with different formatting.
Action: get_order_status(order_id="88213")
Observation: {"error": "order not found"}

... (repeats, incrementing/reformatting the ID, 6 more times) ...
```

This is a hallucinated-argument failure compounding into a near-infinite-loop failure: the model never actually had a real order ID (the user never gave one), so it started guessing, and guessing-then-retrying-on-failure looks superficially like reasonable behavior but never converges. It's caught by two independent guardrails: (1) the iteration cap eventually forces a stop regardless, but that's a late, expensive catch; (2) a *precondition check* is the real fix — the tool (or a wrapper around it) should require the calling agent to have gotten the order ID from a legitimate source (e.g. a prior `lookup_customer_orders(customer_email=...)` call) rather than accepting any string, and the system prompt should instruct the model to ask the user for the order ID rather than guess when it doesn't have one.

**Interview angle**

Q: "What are some concrete ways agents fail, beyond just 'sometimes they're wrong'?"
A: Infinite tool-call loops when the model can't converge, hallucinated tool arguments where it fabricates plausible-looking inputs instead of admitting it doesn't have real data, context overflow from never pruning accumulated tool results, tool misuse where the right tool gets called with wrong arguments or in the wrong situation, and silent partial failures where an incomplete result is treated as success. Each has a specific, different mitigation — iteration/cost caps, argument validation against real data (not just schema shape), context pruning, tighter tool descriptions, and structured error signaling — so naming the mechanism and its fix is what distinguishes a real answer from a vague one.

---

## Evaluating AI Agents

**What it is**

Evaluating an agent is harder than evaluating a single LLM call because success depends on an entire *trajectory* — the sequence of thoughts, tool calls, and observations — not just the final text output; two agents can produce the same correct final answer via very different paths, one efficient and reliable, one that got lucky after three wrong turns. Agent evaluation typically happens at two levels: trajectory-level (did it call the right tools, in a sensible order, with correct arguments, without unnecessary steps?) and outcome-level (is the final answer/action actually correct?). Because "correct" for an agent action often means something changed in the world (a ticket was created, a record was updated), outcome evaluation frequently requires checking real side effects, not just grading text — which is why agent evals lean heavily on sandboxed/simulated environments with ground-truth state, rather than static input/output pairs alone.

**How it works**

- **Outcome evaluation**: did the task actually get completed correctly? For agents with side effects, this means checking the resulting state (did the row actually get updated, did the email actually get sent to the right address) not just parsing the agent's final message.
- **Trajectory evaluation**: was the *path* reasonable — right tools, right order, no redundant or wasted calls, no dangerous near-misses even if it recovered? Often scored by an LLM-as-judge given the full trace, or by exact-match against a reference trajectory for tasks with a known-good path.
- **Step-level/tool-call correctness**: for each individual tool call, were the arguments correct given what was known at that point? This catches hallucinated-argument failures even when the agent's retry logic happened to eventually stumble onto the right answer.
- **Efficiency metrics**: number of LLM calls, tokens, wall-clock time, and dollar cost to complete the task — a "correct but took 40 tool calls and $3" result usually isn't actually acceptable in production.
- **Held-out task suites with sandboxed environments**: because outcomes often mean real side effects, agent benchmarks (e.g. SWE-bench for coding agents, WebArena for web agents) run tasks against a reproducible simulated environment with known ground truth, so success can be checked mechanically (tests pass, correct file state) rather than by grading prose.

**Example**

Evaluating a "book a flight" agent on the task "book the cheapest morning flight to Austin": outcome eval checks the sandboxed booking system's final state — was a flight actually booked, is it a morning flight, was it in fact the cheapest morning option available. Trajectory eval separately checks whether the agent needlessly called `search_flights` five times with near-identical parameters before booking (inefficient but not wrong), or — a real failure — whether it called `book_flight` before ever calling `search_flights`, meaning it booked without checking availability or price at all, which happened to work in this sandbox run but is a dangerous pattern that outcome-only evaluation would have missed entirely.

**Interview angle**

Q: "How do you evaluate an agent, as opposed to evaluating a single LLM response?"
A: You need both outcome and trajectory evaluation — outcome checks whether the task's real side effects are correct (which usually requires a sandboxed environment with ground-truth state, not just text grading), while trajectory evaluation checks whether the *path* taken was sensible: right tools, right order, no dangerous near-misses, no wasted calls. A trajectory that got the right answer by luck after a hallucinated argument or an unsafe out-of-order action is a failure that outcome-only evaluation would miss, so production agent evals track efficiency (calls, tokens, cost) and step-level correctness alongside final-answer correctness.

---

## Framework Landscape

**What it is**

As of this writing, the main frameworks for building agents are LangChain's agent abstractions / LangGraph, AutoGen, CrewAI, OpenAI's Assistants/Responses API with built-in tools, and the Claude Agent SDK. They differ far less in *what* they let you build than in *how explicit the control flow is* and how much of the agentic loop they hide from you versus expose for inspection and control. LangGraph and the Claude Agent SDK favor an explicit state-machine model: you define nodes and edges (or an explicit loop with hooks), and you can inspect, pause, or redirect execution at any point — closer to "you own the loop, the framework gives you primitives." AutoGen and CrewAI favor a role-based, conversational model: you define agents with roles/personas and let them converse with each other to reach a solution, which is faster to prototype but harder to make fully deterministic or debuggable turn-by-turn. Naming this actual axis of difference — explicit graph/state machine versus conversational role-play — is far more convincing in an interview than reciting a list of framework names.

**How it works**

- **LangChain agents / LangGraph**: LangChain provides tool/agent abstractions; LangGraph layers an explicit graph (nodes = steps, edges = transitions, including conditional branches and cycles) on top, letting you define exactly where loops happen, where human-approval checkpoints go, and how state is passed between nodes.
- **AutoGen**: agents defined with roles, having a structured multi-turn conversation with each other (and, optionally, humans) to accomplish a task; control flow emerges from the conversation rather than an explicit graph you drew.
- **CrewAI**: similar role-based multi-agent model to AutoGen, with a more opinionated "crew" abstraction (agents with roles, goals, and a shared task list) aimed at faster setup for common multi-agent patterns.
- **OpenAI Assistants/Responses API**: a hosted agent loop with built-in tools (code execution, file search/retrieval, web browsing in some configurations) where OpenAI's infrastructure manages a good deal of the loop/state for you, trading some control for less boilerplate.
- **Claude Agent SDK**: gives you the primitives (model calls, tool execution, context/session management) to build an explicit agent loop yourself in code, with fine-grained control over what's in context, when to stop, and how to intercept/approve actions — similar philosophy to LangGraph but tied to Claude specifically.
- Choosing between them in practice comes down to: do you need to inspect/pause/redirect execution deterministically (favor explicit-graph frameworks), or is fast prototyping of a multi-role conversation the priority (favor role-based frameworks)?

**Example**

Building a "research and summarize" agent in LangGraph: you'd explicitly define a `search` node, a `read_page` node, a `summarize` node, and a conditional edge that loops back to `search` if the summarize node's output is judged insufficient — every transition is a piece of code you wrote and can unit-test in isolation. Building the same thing in AutoGen: you'd define a "researcher" agent and a "critic" agent and let them converse — the researcher searches and drafts a summary, the critic pushes back if it's thin, and the loop-like behavior emerges from that back-and-forth conversation rather than an explicit edge you drew. Both can produce the same end result; the LangGraph version is more work upfront but the exact control flow is inspectable and testable node-by-node, while the AutoGen version is faster to stand up but harder to guarantee will always follow the same path.

**Interview angle**

Q: "How do the major agent frameworks actually differ from each other?"
A: The real axis is explicit-state-machine versus conversational-role-play, not feature lists. LangGraph and the Claude Agent SDK make you define the control flow explicitly (nodes, edges, loops you can inspect and pause), which costs more upfront design but gives deterministic, debuggable execution. AutoGen and CrewAI let you define agent roles and have them converse, which is faster to prototype but the control flow emerges from the conversation, making it harder to guarantee a fixed path or add deterministic checkpoints. The OpenAI Assistants/Responses API sits further toward "hosted and managed for you," trading control for less boilerplate.

---

## Guardrails Specific to Agents

**What it is**

Agent guardrails are about constraining *actions*, not just text — this is the key distinction from general LLM output guardrails (toxicity filtering, PII redaction, refusal handling), which police what the model *says*. An agent that can call real tools can also take real, sometimes irreversible, actions in the world — delete a file, send a message, issue a refund, execute a trade — so the risk surface extends well beyond "did it generate something offensive." The standard mitigations are: sandboxed tool execution (the agent's tools run in an environment where a mistake can't cause real damage, e.g. a scratch filesystem or a staging API instead of production); allow-lists of permitted actions (the agent can only call tools explicitly whitelisted for its role, rather than anything technically reachable); mandatory human approval before irreversible or high-stakes actions (file deletion, payments, sending external communications) — the agent proposes, a human confirms, before execution; and cost/step budgets (hard ceilings on how many tool calls or how much money a single agent run can consume, independent of whether it's "still making progress").

**How it works**

- **Sandboxing**: run tool execution against non-production resources (staging environment, read-only replica, scratch directory) wherever the task allows, so a wrong action's blast radius is contained.
- **Allow-listing**: the agent's available tool set is scoped per role/context — a customer-support agent might have `read_order`, `issue_refund_under_$50`, but not `delete_customer_account`, even if such a tool technically exists in the codebase.
- **Human-in-the-loop approval**: for actions above a defined risk threshold (irreversible, above a dollar amount, affecting external parties), the agent's proposed action is surfaced to a human for explicit confirmation before it executes — the agent still does the reasoning and drafting, a human gates the actual trigger-pull.
- **Cost/step budgets**: independent of task-completion logic, a hard ceiling (max tool calls, max tokens, max dollars) stops a run regardless of whether the agent believes it's still making progress — this overlaps with the agentic loop's own stop conditions but is worth calling out separately as a *safety* control, not just a cost control.
- These sit alongside, not instead of, general LLM safety guardrails (content filtering, prompt-injection defenses, PII handling) — an agent needs both layers, because the action layer and the text layer fail in different ways and require different fixes.

**Example**

A "manage my calendar" agent with a `delete_event` tool: without guardrails, a prompt-injected email ("please delete all my meetings this week") or a simple misunderstanding could wipe a user's calendar with no recovery path. With guardrails: `delete_event` is allow-listed only for events the agent itself created in this session (not arbitrary pre-existing events), any deletion of an event with more than one attendee requires explicit user confirmation before executing, and the action runs against a versioned calendar API where deletions are soft (recoverable for 30 days) rather than hard — sandboxing the blast radius of a mistake even if every other guardrail somehow fails.

**Interview angle**

Q: "How are agent guardrails different from the guardrails you'd put on any LLM output?"
A: General LLM guardrails police *text* — toxicity, PII leakage, refusals for disallowed content. Agent guardrails police *actions* — sandboxed execution so mistakes are contained, allow-lists so an agent can only reach tools appropriate to its role, mandatory human approval before irreversible or high-stakes actions, and hard cost/step budgets independent of the model's own sense of progress. Both layers are necessary for an agent, since it can fail at the text level and the action level independently — see [08-guardrails-and-llm-safety-concepts.md](08-guardrails-and-llm-safety-concepts.md) for the broader safety taxonomy this fits into.
