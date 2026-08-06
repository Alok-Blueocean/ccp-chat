# LangChain Concepts

General LangChain knowledge. Note: this repo does **not** use LangChain for its retrieval pipeline — it uses **LlamaIndex** (see [01-retrieval-and-rag.md](01-retrieval-and-rag.md)) plus raw `litellm`/`openai` calls, and only pulls in `langchain-openai` wrappers for RAGAS evaluation (`evaluation/evaluate_testset.py`, `evaluation/evaluate_langfuse.py` — RAGAS's metrics take a LangChain `ChatOpenAI`/`OpenAIEmbeddings` instance as their LLM/embedding backend). Knowing *that* distinction is itself a good interview answer: it shows you picked a retrieval-first framework for retrieval, rather than defaulting to "the popular one" for a job it isn't specialized for.

Everything below is general LangChain knowledge (not this repo's code) — worth knowing cold for interviews since LangChain is the framework most interviewers assume as a baseline vocabulary, even when a project (like this one) doesn't use it directly.

## Runnable & LCEL

**What it is**: `Runnable` is the single interface every LangChain component implements — prompts, chat models, retrievers, output parsers, and even plain Python functions (via `RunnableLambda`). Because they all expose the same methods (`invoke`, `batch`, `stream`, and their `a`-prefixed async twins), any two Runnables can be composed with the `|` operator into a new Runnable. LCEL (LangChain Expression Language) is just the name for this pipe-composition style: `prompt | llm | parser` builds a pipeline where the output of each step becomes the input of the next, and the resulting object is itself a Runnable you can invoke, stream, or batch. This is the backbone of essentially all modern (post-2023) LangChain code — the older `Chain` subclasses are legacy by comparison.

**How it works**:
- `.invoke(input)` runs the pipeline once and returns the final output.
- `.batch([input1, input2, ...])` runs multiple inputs, parallelized under the hood where possible.
- `.stream(input)` yields incremental chunks as they're produced (token-by-token for an LLM step).
- `RunnableParallel` (or a plain dict literal in a chain) fans one input out to multiple Runnables concurrently and collects results into a dict.
- `RunnableLambda` wraps an arbitrary Python function so it can sit inside a pipe.
- `RunnablePassthrough` forwards the input unchanged, commonly used to keep the original question alongside a retrieved-context branch.
- `RunnableBranch` adds conditional routing (if/elif/else) between different sub-chains based on the input.
- Every Runnable automatically gets `.with_retry()`, `.with_fallbacks()`, and `.with_config()` (e.g., to attach tags/callbacks) for free.

**Example**:
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
parser = StrOutputParser()

# Basic LCEL pipe: prompt -> model -> parser
chain = prompt | llm | parser
print(chain.invoke({"topic": "LCEL"}))

# Fan-out with RunnableParallel + RunnablePassthrough
enriched = RunnableParallel(
    answer=chain,
    original_topic=RunnablePassthrough(),
)
print(enriched.invoke({"topic": "Runnables"}))

# Streaming: yields tokens as the model produces them
for chunk in chain.stream({"topic": "streaming"}):
    print(chunk, end="", flush=True)
```

**Interview angle**:
Q: What's the difference between LCEL and the old `LLMChain`-style chains?
A: LCEL isn't a separate feature bolted onto chains — it's the unifying interface underneath everything. Every component (prompt, model, retriever, parser) implements the same `Runnable` contract, so `|` is just composing functions with a shared calling convention. `LLMChain` and friends were purpose-built wrapper classes with their own ad hoc APIs; LCEL replaced the need for most of them because you get batching, streaming, async, retries, and fallbacks automatically on any pipeline you build, without a bespoke class for each shape of chain.

## Chains

**What it is**: "Chains" is the pre-LCEL abstraction for composing a sequence of calls (often LLM calls) into a single reusable object — `LLMChain`, `SequentialChain`, `TransformChain`, etc. Conceptually a chain is a fixed, non-branching (or simply-branching) pipeline: input goes in one end, a series of steps run in order, output comes out the other end. LCEL now covers almost all of what these classes did, and is the recommended way to build new chains, but the *concept* of "chain" — a linear DAG of steps with no loops — still matters because it defines the boundary of what chains, old or new, can express.

**How it works**:
- `SequentialChain` ran multiple chains in order, threading named outputs of one into the named inputs of the next.
- `TransformChain` let you inject an arbitrary Python transformation step between LLM calls.
- Legacy chains stored configuration (prompt, llm, output key) as constructor arguments and exposed a single `.run()`/`.invoke()` entry point, hiding the pipeline shape from the caller.
- Because chains are a DAG, they **cannot loop** — there's no built-in way to say "keep retrying / reasoning until a condition is met" inside a chain itself.
- Some legacy classes (`RetrievalQA`, `ConversationalRetrievalChain`) bundled a whole RAG pattern (retrieve → stuff into prompt → generate) into one convenience class; modern code builds the same pattern explicitly with LCEL for clarity and control.

**Example**:
```python
# Modern equivalent of "RetrievalQA" built explicitly with LCEL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = rag_chain.invoke("What is LCEL?")
```

**Interview angle**:
Q: Why did LangChain move away from `Chain` subclasses toward LCEL?
A: The class-based chains each needed their own bespoke API, made it hard to introspect or modify the pipeline shape, and — most importantly — were strictly acyclic. Once people wanted agent-like behavior (loop until the task is done, revisit earlier steps, branch dynamically on model output), a DAG-only abstraction couldn't express it. That gap is exactly what motivated LangGraph, which models the pipeline as an explicit graph with edges that can cycle back (see [10-langgraph-concepts.md](10-langgraph-concepts.md)).

## Prompt templates

**What it is**: `PromptTemplate` (plain string) and `ChatPromptTemplate` (structured message list) are parameterized prompt objects with named variable slots (`{variable}`) that get filled in at call time via `.invoke({...})` or `.format(...)`. Instead of hand-building prompt strings with f-strings scattered through the codebase, templates centralize the prompt's structure, let you reuse it across many inputs, and — for chat templates — encode the conversation's role structure (system / human / AI / tool) as first-class objects rather than string concatenation.

**How it works**:
- `ChatPromptTemplate.from_messages([...])` takes a list of `(role, template_string)` tuples or message objects, one per turn/role.
- `MessagesPlaceholder("history")` reserves a slot where a list of prior messages (e.g., from memory or a `messages` key) gets spliced in at render time — essential for multi-turn chat prompts.
- `.partial(...)` binds some variables ahead of time, returning a new template that only needs the remaining ones — useful for injecting a fixed system persona once and reusing the template per-request with just the user's question.
- Few-shot prompting is supported via `FewShotPromptTemplate` / `FewShotChatMessagePromptTemplate`, which render a list of example input/output pairs into the prompt before the real query, optionally chosen dynamically by a `ExampleSelector` (e.g., semantic similarity to the current input).
- Prompt templates are Runnables themselves, so they compose directly with `|` into LCEL pipelines — no adapter needed.

**Example**:
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful support agent for {product}."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# .partial binds `product` once; the rest is filled per-request
support_prompt = prompt.partial(product="Acme Widgets")

messages = support_prompt.invoke({
    "chat_history": [("human", "My widget won't turn on."),
                      ("ai", "Have you tried charging it?")],
    "question": "I did, still nothing.",
}).to_messages()
```

**Interview angle**:
Q: Why use `ChatPromptTemplate` instead of building the prompt string yourself?
A: Two reasons: role structure and reuse. Chat models expect a list of role-tagged messages (system/human/ai), and `ChatPromptTemplate` encodes that natively instead of you hand-formatting strings and hoping the delimiters match what the model expects. It also decouples the prompt's *shape* from any single call site — the same template can be partially bound, reused across a batch of requests, and composed into an LCEL pipe — versus raw f-strings, which is exactly what this repo does with `CHAT_SYSTEM` in `llm_service.py`. That's a legitimate simpler alternative when you don't need multi-template reuse or few-shot injection; templates start paying for themselves once you do.

## Output parsers / structured output

**What it is**: Output parsers convert a model's raw text (or native tool-call payload) into a typed, usable object instead of leaving the caller to regex-scrape a string. `StrOutputParser` is the trivial case (just extract the text). `PydanticOutputParser` (and its modern counterpart, binding a Pydantic model via `.with_structured_output()`) coerces the response into a validated Pydantic object, either by instructing the model to emit JSON matching a schema (and then parsing/validating it) or, on providers that support it, by using the provider's native structured-output / tool-calling mode so the model is constrained to valid JSON at generation time rather than being asked nicely.

**How it works**:
- `parser.get_format_instructions()` returns schema-description text you inject into the prompt so the model knows the exact shape to produce (field names, types) when you're doing prompt-based parsing.
- `parser.parse(raw_text)` (or `parser.invoke(raw_text)` in LCEL form) turns the raw string into the typed object, raising a parsing error you can catch and retry on (`OutputFixingParser` / `RetryOutputParser` wrap this retry logic automatically).
- `llm.with_structured_output(PydanticModel)` is the modern, preferred path on providers with native structured-output/function-calling support: it returns a new Runnable that outputs an already-validated Pydantic instance directly, no manual JSON parsing or format instructions needed.
- Under the hood, native structured output is usually implemented via the provider's function/tool-calling API — the "function" is really just the schema, and the "call" is really the structured answer.
- Parsing failures are a real failure mode to design for: a chain built on `PydanticOutputParser` needs a fallback/retry strategy for malformed output; `with_structured_output` reduces but doesn't eliminate this (schema violations from smaller/uncooperative models still happen).

**Example**:
```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Ticket(BaseModel):
    category: str = Field(description="one of: billing, bug, feature_request")
    priority: int = Field(description="1 (low) to 5 (urgent)")
    summary: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(Ticket)

result = structured_llm.invoke(
    "My invoice charged me twice this month, please fix ASAP."
)
print(result.category, result.priority, result.summary)
# -> "billing" 5 "Customer double-charged on invoice, needs urgent fix"
```

**Interview angle**:
Q: What's the difference between `PydanticOutputParser` and `with_structured_output()`, and when would you still need the former?
A: `PydanticOutputParser` is a *prompt-engineering* trick — it injects schema instructions into the prompt and then parses whatever text comes back, which can fail if the model doesn't comply exactly. `with_structured_output()` instead leans on the provider's native constrained-decoding / function-calling machinery, so the output is far more reliably valid JSON, at the cost of requiring provider support. You'd still reach for the parser-based approach with providers or local models that don't expose a native structured-output API — the "guardrail" framing from [03-evaluation-and-observability.md](03-evaluation-and-observability.md) applies either way: never trust raw LLM text to already be your target schema.

## Retrievers & VectorStore abstraction

**What it is**: `VectorStore` is a common interface (`add_documents`, `similarity_search`, `as_retriever()`) that wraps many different vector database backends — Chroma, Pinecone, Qdrant, pgvector, FAISS, and dozens more — behind identical calling code. `Retriever` is a level up: the general interface (`.invoke(query) -> list[Document]`) for *anything* that can turn a query into relevant documents, of which a vector store's retriever is just one implementation — you can equally have a keyword/BM25 retriever, a web-search retriever, or a multi-retriever ensemble, all exposing the same `.invoke()` contract so they drop into an LCEL chain interchangeably.

**How it works**:
- `vectorstore.as_retriever(search_kwargs={"k": 5})` turns a vector store into a `Retriever` Runnable with a fixed top-k and (optionally) score threshold or metadata filter.
- `MultiQueryRetriever` generates several paraphrased versions of the input query via an LLM, retrieves for each, and de-duplicates/merges the results — a mitigation for queries that are phrased differently than the indexed text.
- `ContextualCompressionRetriever` wraps a base retriever with a "compressor" step (an LLM-based extractor or a re-ranker) that trims each retrieved document down to just the relevant portion before returning it — cheap way to reduce prompt-stuffing noise.
- `EnsembleRetriever` runs multiple retrievers (e.g., a dense vector retriever + a BM25 keyword retriever) in parallel and merges/re-ranks their results — the LangChain-native version of hybrid search.
- In theory, swapping the vector DB backend means changing only the `VectorStore` implementation you instantiate — call sites (`.as_retriever()`, `.invoke()`) stay the same; in practice, backend-specific advanced features (custom fusion, named/multi-vectors, filtered hybrid search) usually require backend-specific code anyway, so the abstraction is clean for basic similarity search and leakier for advanced retrieval.

**Example**:
```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

docs = [
    Document(page_content="LCEL composes Runnables with the | operator."),
    Document(page_content="Chroma is a local, open-source vector store."),
]

vectorstore = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

results = retriever.invoke("How do you compose Runnables?")
for doc in results:
    print(doc.page_content)
```

**Interview angle**:
Q: If LangChain's retriever abstraction is backend-agnostic, why would a RAG system still pick a specific vector DB and write custom retrieval code, like this repo's hybrid RRF fusion in LlamaIndex?
A: The `Retriever` interface guarantees a uniform *call signature*, not uniform *capability* — `.invoke(query) -> documents` says nothing about whether the backend supports multiple named vectors, server-side hybrid fusion, or metadata-filtered ANN search efficiently. The moment a retrieval strategy needs a feature the generic interface didn't anticipate (e.g., reciprocal rank fusion across dense + sparse named vectors, which is exactly what this repo's Qdrant-based pipeline does), you end up writing backend-specific code below the abstraction anyway, and the "swap the vector DB freely" pitch only holds for the common-denominator case of plain top-k similarity search.

## Memory

**What it is**: Memory components persist conversational state across turns so a chain/agent "remembers" earlier parts of the conversation without the caller re-passing the entire history manually every time. Classic examples: `ConversationBufferMemory` (keep every message verbatim), `ConversationSummaryMemory` (periodically summarize older turns to bound token growth), and `ConversationBufferWindowMemory` (keep only the last N turns). These classes are largely legacy in current LangChain guidance — newer code manages history either as an explicit list of messages passed into a `MessagesPlaceholder`, or via LangGraph's checkpointer, which persists the entire graph state (including message history) keyed by a thread/session id.

**How it works**:
- Legacy memory objects expose `.load_memory_variables()` (read prior state into the prompt) and `.save_context()` (append the latest turn) hooks that a chain calls automatically before/after each run.
- `ConversationSummaryMemory` uses an LLM call to compress older history into a running summary, trading fidelity for a bounded prompt size as conversations grow long.
- All of these classes are in-process/in-memory by default — restart the process and history is gone, unless you back them with a persistent store (`RedisChatMessageHistory`, `PostgresChatMessageHistory`, etc.).
- The modern pattern threads `chat_history` explicitly as a list of messages (often via `RunnableWithMessageHistory`, which wraps a chain and auto-loads/saves history per `session_id` from a pluggable store) rather than using an opaque "Memory" object baked into the chain.
- LangGraph's checkpointer is the current recommended path for anything beyond trivial chat history: it persists the *entire* graph state (not just messages) after every step, keyed by thread id, giving you durable, resumable, inspectable conversation state for free.

**Example**:
```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}  # session_id -> InMemoryChatMessageHistory; swap for Redis/Postgres in prod

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

chain_with_history.invoke(
    {"question": "My name is Alok."},
    config={"configurable": {"session_id": "user-42"}},
)
chain_with_history.invoke(
    {"question": "What's my name?"},
    config={"configurable": {"session_id": "user-42"}},
)  # -> recalls "Alok" because history was auto-loaded for this session_id
```

**Interview angle**:
Q: Is "LangChain memory" the same thing as durable chat history storage, like a Postgres-backed chat table?
A: No — they solve different layers of the problem. LangChain memory (classic or `RunnableWithMessageHistory`) is about *shaping what goes into the prompt* — deciding whether to include full history, a summary, or the last N turns. Durability — surviving a process restart, being queryable by other services, supporting retention/deletion policies — is a database concern that the memory abstraction explicitly delegates to whatever store you plug in. A production system, like this repo's Postgres-backed `chat_memory.py`, typically owns the persistence layer directly rather than trusting an in-process memory object with something that needs to survive restarts.

## Tools & Agents

**What it is**: A "tool" is a typed, named, described callable — a schema plus a function — that an LLM can choose to invoke instead of answering directly, via the model's native function/tool-calling capability. An "agent" is the runtime loop that repeatedly (1) asks the model what to do next given the conversation and available tools, (2) executes any tool call the model requests, (3) feeds the result back in, and (4) repeats until the model decides to answer instead of calling another tool — the ReAct-style reasoning/acting loop. The classic implementation was `AgentExecutor`, a mostly-opaque loop you configure with a model, a tool list, and a prompt; you don't see or control what happens between "give it the tools" and "get the final answer."

**How it works**:
- `@tool` decorates a plain Python function, using its type hints and docstring to auto-generate the schema (name, description, parameter types) the model sees when deciding whether to call it.
- `llm.bind_tools([tool1, tool2])` returns a model Runnable that, on invocation, may return either a normal text response or a `tool_calls` list — the model itself decides.
- `AgentExecutor` wraps this into a loop: call the model, if it requested tool calls, execute them (`ToolNode`-equivalent logic) and append `ToolMessage` results, then call the model again — repeat until a plain text answer comes back, or a max-iteration/time limit is hit.
- The classic executor is a black box: you can't easily pause mid-loop, inspect intermediate state, add a human-approval gate, or resume after a crash — you get the whole loop or nothing.
- This limitation is the direct, explicit motivation for LangGraph (see [10-langgraph-concepts.md](10-langgraph-concepts.md)): model the same loop as an explicit graph (model node → conditional edge → tool node → back to model node) so every step is a first-class, inspectable, interruptible, checkpointable state transition instead of hidden control flow inside one class.

**Example**:
```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a given city name."""
    return f"It's 22C and sunny in {city}."

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [get_weather], prompt)
executor = AgentExecutor(agent=agent, tools=[get_weather], verbose=True)

result = executor.invoke({"input": "Should I bring an umbrella in Paris today?"})
print(result["output"])
```

**Interview angle**:
Q: Why did LangGraph replace `AgentExecutor` rather than just extending it?
A: `AgentExecutor`'s loop — call model, run any requested tools, repeat — is fixed and internal to the class; extending it to support things like a human-in-the-loop approval step before a risky tool call, resuming after a process crash, or running two tool calls concurrently with different fan-in logic means fighting the class's assumptions rather than composing with them. LangGraph solves this by making every step of that same loop an explicit node and edge in a graph you define, so pausing, inspecting, checkpointing, and branching are native graph operations instead of workarounds bolted onto a closed loop.

## Callbacks & tracing

**What it is**: LangChain's callback system is a set of lifecycle hooks (`on_llm_start`, `on_llm_end`, `on_chain_start`, `on_tool_end`, `on_retriever_end`, etc.) that fire at every step of a Runnable's execution, letting external code observe — without modifying — a run. Any object implementing the `BaseCallbackHandler` interface can subscribe to these events; you can attach one via `config={"callbacks": [handler]}` on any `.invoke()` call, or globally via environment/config. LangSmith, LangChain's own tracing/observability product, is built entirely on top of this callback system — every LLM call, tool call, and intermediate chain step gets logged as a nested span in a trace, which is exactly the same conceptual job Langfuse's `@observe` decorators do in this repo (see [03-evaluation-and-observability.md](03-evaluation-and-observability.md)).

**How it works**:
- Handlers receive structured event data (inputs, outputs, timing, token usage, errors) at each callback, and can do anything with it — log to stdout, push to a tracing backend, compute running cost totals, etc.
- Setting `LANGCHAIN_TRACING_V2=true` plus a `LANGCHAIN_API_KEY` env var auto-enables LangSmith tracing for every LangChain call in the process, with zero code changes — this is the most common way tracing gets turned on in practice.
- Traces in LangSmith are hierarchical: a top-level chain/agent run contains nested spans for each LLM call, tool call, and retriever call inside it, mirroring the actual call tree rather than a flat log.
- Callbacks propagate automatically through an LCEL pipeline — attaching a handler at the top-level `.invoke()` call means it also fires for every nested Runnable inside the chain, without manually wiring each one.
- The same category of tool exists across the ecosystem under different names — LangSmith, Langfuse, Helicone, Arize Phoenix — all doing "capture a nested trace of every LLM/tool/retriever call, with timing/cost/token metadata, for debugging and evaluation."

**Example**:
```python
from langchain_core.callbacks import BaseCallbackHandler

class SimpleLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[LLM start] prompt: {prompts[0][:60]}...")

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        print(f"[LLM end] tokens used: {usage}")

    def on_tool_end(self, output, **kwargs):
        print(f"[Tool end] output: {output}")

chain = prompt | llm
chain.invoke({"topic": "callbacks"}, config={"callbacks": [SimpleLogger()]})
```

**Interview angle**:
Q: How does LangSmith actually capture a trace — is it special-cased into the model calls, or something more general?
A: It's built entirely on the general-purpose callback system, not a special code path — LangSmith is just a `BaseCallbackHandler` (attached automatically once tracing env vars are set) that listens to the same `on_llm_start`/`on_chain_end`/etc. events any custom handler could subscribe to, and ships them to LangSmith's backend as nested spans. That's a useful thing to point out because it means the observability story generalizes: the same hooks work for a custom logger, cost tracker, or a different vendor's tracer like Langfuse, which is the tool this repo actually uses for the same purpose.

## Document loaders & text splitters

**What it is**: Document loaders (`PyPDFLoader`, `WebBaseLoader`, `CSVLoader`, `UnstructuredFileLoader`, and dozens more) ingest raw source material — PDFs, HTML pages, CSVs, Notion exports, Slack dumps — into a common `Document` object: a `page_content` string plus a `metadata` dict (source path, page number, etc.). Text splitters then break those (often too-long) documents into smaller chunks sized appropriately for embedding and retrieval — `RecursiveCharacterTextSplitter` (split on a prioritized list of separators — paragraphs, then sentences, then words — falling back progressively until chunks fit the size limit), `TokenTextSplitter` (split by actual token count rather than characters), and semantic/markdown/code-aware splitters that respect document structure.

**How it works**:
- Loaders normalize source-format quirks (PDF page boundaries, HTML tag noise, CSV rows) behind one `.load() -> list[Document]` call, so downstream indexing code doesn't need per-format branches.
- `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` tries splitting on `"\n\n"` first, then `"\n"`, then `" "`, then characters — recursing only as needed to hit the target chunk size, which tends to preserve paragraph/sentence boundaries better than a naive fixed-width cut.
- `chunk_overlap` deliberately duplicates a slice of text between adjacent chunks so context isn't lost right at a chunk boundary (e.g., a sentence split exactly in half).
- Metadata set at load time (source file, page number) propagates through splitting, so each final chunk still traces back to exactly where it came from — essential for citations.
- The chunking-strategy decision (fixed-size vs. semantic vs. hierarchical/parent-child) is the same tradeoff this repo makes with LlamaIndex's `HierarchicalNodeParser` — different library, identical underlying problem: retrieve on small precise chunks, but generate with enough surrounding context.

**Example**:
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("handbook.pdf")
docs = loader.load()  # one Document per page, metadata={"source": ..., "page": ...}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"{len(docs)} pages -> {len(chunks)} chunks")
print(chunks[0].metadata)  # {"source": "handbook.pdf", "page": 0}
```

**Interview angle**:
Q: Why use `chunk_overlap` at all — doesn't it just duplicate content and waste index space?
A: Yes, it trades some index bloat for retrieval robustness at chunk boundaries. Without overlap, a fact that happens to be split exactly across two chunks (e.g., "the deadline is... [chunk boundary] ...March 15th") can end up not fully retrievable by either chunk in isolation. A modest overlap (10-20% of chunk size is typical) means boundary-straddling content is fully present in at least one chunk, at the cost of some duplicated embeddings — the same "recall vs. index size" tradeoff shows up in [01-retrieval-and-rag.md](01-retrieval-and-rag.md)'s chunking discussion, just from the LangChain side of the fence.

## LangChain vs. LlamaIndex

**What it is**: Both are general-purpose LLM application frameworks with substantial feature overlap (both have document loaders, retrievers, chat models, agents), but they grew from different centers of gravity. LangChain started general-purpose — chains, tools, agents — and retrieval was one feature among many. LlamaIndex started retrieval/indexing-first — its `Node`, `NodeParser`, index (vector/keyword/tree/graph), and `NodeWithScore` retriever primitives are more mature and more configurable specifically for RAG, because that was the framework's original and still-primary focus. In practice today the gap has narrowed (LangChain's retrieval story is solid; LlamaIndex now also ships agent/workflow abstractions), so the choice is often about which primitives fit the team's specific pipeline shape best, plus ecosystem/integration fit.

**How it works**:
- LlamaIndex's `NodeParser` family (including `HierarchicalNodeParser`) natively supports parent-child chunk relationships — retrieve on a small precise child chunk, but pull the larger parent chunk into the generation context — as a first-class indexing concept, not something bolted on.
- LlamaIndex's retriever results (`NodeWithScore`) carry a similarity score and rich node relationships (parent/child/prev/next) by default; LangChain's plain `Document` return type carries `page_content` + `metadata` only, with anything more structured left to the caller to build.
- LangChain's comparative strength is breadth outside retrieval: LCEL composition, the tool/agent ecosystem, and (via LangGraph) first-class support for stateful, cyclic, multi-agent workflows — areas where LlamaIndex's newer "Workflows" abstraction is catching up but has less maturity/adoption.
- Nothing stops using both in the same codebase for what each is strongest at — which is exactly this repo's approach: LlamaIndex owns indexing/retrieval, and `langchain-openai`'s `ChatOpenAI`/`OpenAIEmbeddings` wrapper classes are pulled in purely because RAGAS (the evaluation library) expects a LangChain-shaped LLM/embeddings object as its evaluator backend, not because any retrieval logic runs through LangChain.
- Neither framework is "more correct" in the abstract — the decision criterion is which one's primitives most directly match the pipeline you're building, not brand popularity.

**Example** (the actual reason this repo touches LangChain at all — RAGAS needs a LangChain-shaped model object as its judge/embedding backend, even though retrieval itself is pure LlamaIndex):
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator_embeddings = OpenAIEmbeddings()

results = evaluate(
    dataset=my_ragas_dataset,          # built from this repo's LlamaIndex RAG outputs
    metrics=[faithfulness, answer_relevancy],
    llm=evaluator_llm,                 # RAGAS's metric implementations expect
    embeddings=evaluator_embeddings,   # this exact LangChain interface as their backend
)
```

**Interview angle**:
Q: This repo uses LlamaIndex for RAG but imports `langchain-openai` — isn't that contradictory?
A: No — it's two unrelated jobs that happen to share a dependency name. LlamaIndex handles 100% of the actual retrieval pipeline (hierarchical chunking, hybrid RRF fusion, node retrieval). `langchain-openai`'s `ChatOpenAI`/`OpenAIEmbeddings` classes show up only because RAGAS, the evaluation library, standardized its metric implementations (faithfulness, answer relevancy, etc.) around LangChain's LLM/embeddings interface as the pluggable "judge" backend — so using RAGAS at all means handing it a LangChain-shaped object, regardless of what framework generated the answers being evaluated. It's a good example of a dependency existing for interoperability with a *third* library, not because the retrieval architecture is secretly LangChain-based.
