"""
agent_tools.py — LLM-callable tools over the public training databank.

Used to predict whether a target model answers an item correctly. The agent
passes its public database directly to the dispatcher; retrieval needs no
target identifiers or temporary corpus-configuration files.

Each tool carries updated_belief: the estimate BEFORE this retrieval executes.
The agent reads the retrieved evidence when making its next estimate.
"""

from agent.tools import _BELIEF_SCHEMA

from .databank import MeasurementDB


def _tool(name, description, props, required):
    props = dict(props)
    props["updated_belief"] = _BELIEF_SCHEMA
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required + ["updated_belief"],
            },
        },
    }


_BID = {
    "type": "string",
    "description": "Benchmark id (as returned by search_benchmarks)",
}

MEASUREMENT_TOOLS = [
    _tool(
        "semantic_search_items",
        "Semantic search over ALL items in the databank corpus at once, by "
        "meaning rather than keywords (embedding cosine similarity). "
        "Phrase whatever you want to find as a natural-language question or "
        "description — or paste the target item's text itself — and get the "
        "most similar benchmark items with their benchmark, similarity, and "
        "mean score across models (difficulty). Use this to discover comparable "
        "tasks, then inspect them with get_benchmark, get_scores, or get_item.",
        {
            "query": {
                "type": "string",
                "description": "Natural-language question, task description, "
                "or item text to find related evidence for",
            },
            "k": {
                "type": "integer",
                "description": "Number of results (1-20, default 8)",
            },
            "benchmark_id": {
                "type": "string",
                "description": "Optional: restrict the search to this one benchmark",
            },
        },
        ["query"],
    ),
    _tool(
        "search_benchmarks",
        "Search the public training benchmarks by keyword (over name, "
        "description, domain, modality). Use this to discover benchmarks similar "
        "to the item being predicted — e.g. same task type, domain, or skill. "
        "Returns brief benchmark cards with ids for use in other tools.",
        {
            "query": {
                "type": "string",
                "description": "Keywords, e.g. 'code generation agentic' or 'medical QA'",
            }
        },
        ["query"],
    ),
    _tool(
        "get_benchmark",
        "Get the full metadata card of one benchmark: description, test "
        "conditions, number of items/models/trials, mean score across all "
        "models (saturation), and which data tables are available.",
        {"benchmark_id": _BID},
        ["benchmark_id"],
    ),
    _tool(
        "find_model",
        "Find models in the databank by (partial) name and list the benchmarks "
        "each was evaluated on, newest first. Use this to locate the target "
        "model's evaluation record, or to find comparison models.",
        {
            "model_name": {
                "type": "string",
                "description": "Model name or substring, e.g. 'GPT-4o' or 'Claude'",
            }
        },
        ["model_name"],
    ),
    _tool(
        "get_scores",
        "Get aggregate scores on a benchmark: mean score per model x test "
        "condition (a leaderboard). Optionally filter to one model and/or one "
        "condition. This is the main quantitative evidence source — use it to "
        "see how the target model and its peers perform on related benchmarks.",
        {
            "benchmark_id": _BID,
            "model": {
                "type": "string",
                "description": "Optional model name filter (substring match)",
            },
            "test_condition": {
                "type": "string",
                "description": "Optional exact test-condition filter",
            },
        },
        ["benchmark_id"],
    ),
    _tool(
        "search_items",
        "Keyword-search the items (questions) inside one benchmark. Returns "
        "matching items with a content preview and their mean score across all "
        "models (difficulty). Use this to check whether a benchmark's items "
        "really resemble the item being predicted.",
        {
            "benchmark_id": _BID,
            "query": {
                "type": "string",
                "description": "Keywords to match item content",
            },
        },
        ["benchmark_id", "query"],
    ),
    _tool(
        "get_item",
        "Read one item in full (content, correct answer if released, difficulty), "
        "or a small random sample of items from a benchmark. Optionally include "
        "a specific model's per-trial scores on that item.",
        {
            "benchmark_id": _BID,
            "item_id": {
                "type": "string",
                "description": "Item id (from search_items); omit to sample randomly",
            },
            "sample": {
                "type": "integer",
                "description": "Number of random items (1-5) when item_id is omitted",
            },
            "model": {
                "type": "string",
                "description": "Optional model name: include its "
                "per-trial scores on the item",
            },
        },
        ["benchmark_id"],
    ),
    _tool(
        "get_trace",
        "Read a model's actual output/reasoning trace on one item of a "
        "benchmark, when trace data exists. Use this to inspect HOW a model "
        "succeeds or fails on items similar to the one being predicted.",
        {
            "benchmark_id": _BID,
            "model": {"type": "string", "description": "Model name (substring match)"},
            "item_id": {"type": "string", "description": "Item id (from search_items)"},
        },
        ["benchmark_id", "model", "item_id"],
    ),
]

MEASUREMENT_TOOL_NAMES = {t["function"]["name"] for t in MEASUREMENT_TOOLS}


def get_measurement_tools():
    """Offer semantic search only when a local embeddings payload is present."""
    from .dataroot import emb_root

    has_embeddings = emb_root() is not None
    return [
        tool
        for tool in MEASUREMENT_TOOLS
        if has_embeddings or tool["function"]["name"] != "semantic_search_items"
    ]


def dispatch_measurement_tool(name: str, args: dict, db: MeasurementDB) -> str:
    """Execute a retrieval. Missing matches return text; execution errors raise."""
    if name == "semantic_search_items":
        return db.semantic_search_items(
            args["query"], k=args.get("k", 8), benchmark_id=args.get("benchmark_id", "")
        )
    if name == "search_benchmarks":
        return db.search_benchmarks(args["query"])
    if name == "get_benchmark":
        return db.get_benchmark(args["benchmark_id"])
    if name == "find_model":
        return db.find_model(args["model_name"])
    if name == "get_scores":
        return db.get_scores(
            args["benchmark_id"],
            model=args.get("model", ""),
            test_condition=args.get("test_condition", ""),
        )
    if name == "search_items":
        return db.search_items(args["benchmark_id"], args["query"])
    if name == "get_item":
        return db.get_item(
            args["benchmark_id"],
            item_id=args.get("item_id", ""),
            sample=args.get("sample", 0),
            model=args.get("model", ""),
        )
    if name == "get_trace":
        return db.get_trace(args["benchmark_id"], args["model"], args["item_id"])
    raise ValueError(f"unknown retrieval tool: {name!r}")
