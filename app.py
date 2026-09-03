"""Streamlit UI for the SEC 10-K RAG assistant.

Every interaction (question, answer, retrieval config, latency) and any thumbs
up/down feedback is logged to Postgres (monitoring/db.py) for the Grafana
dashboard (monitoring/grafana/) — this is what satisfies the DTC rubric's
monitoring criterion (user feedback collected + a dashboard with charts).

Run:
    uv run streamlit run app.py
"""

import concurrent.futures
import time

import streamlit as st
from dotenv import load_dotenv

from monitoring import db
from rag.agent import MAX_ITERATIONS, agentic_answer
from rag.fields import SECTOR
from rag.generate import generate_answer
from rag.judge import judge_answer_quality
from rag.pipeline import retrieve

load_dotenv(override=True)

st.set_page_config(page_title="SEC 10-K RAG Assistant", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner=False)
def _init_db() -> bool:
    db.init_db()
    return True


_init_db()

st.title("📊 SEC 10-K RAG Assistant")
st.caption(
    "Retrieval-Augmented Generation over SEC 10-K annual reports "
    "(tech: GOOGL, MSFT, NVDA · banks: JPM, GS, BAC). "
    "Answers are grounded in retrieved passages and cite their sources."
)


def _run_traditional(query: str, k: int, filter_dict: dict | None, method: str, use_rerank: bool, use_rewrite: bool) -> dict:
    start = time.monotonic()
    hits = retrieve(query, k=k, filter_dict=filter_dict, method=method, use_rerank=use_rerank, use_query_rewrite=use_rewrite)
    answer = generate_answer(query, hits)
    return {
        "answer": answer, "hits": hits, "tool_calls": None,
        "db_method": method, "db_use_rerank": use_rerank, "db_use_rewrite": use_rewrite,
        "db_filter_dict": filter_dict, "n_tool_calls": None,
        "latency_ms": int((time.monotonic() - start) * 1000),
    }


def _run_agentic(query: str, max_iterations: int) -> dict:
    start = time.monotonic()
    result = agentic_answer(query, max_iterations=max_iterations)
    return {
        "answer": result["answer"], "hits": result["hits"], "tool_calls": result["tool_calls"],
        "db_method": "dense", "db_use_rerank": True, "db_use_rewrite": False,
        "db_filter_dict": None, "n_tool_calls": len(result["tool_calls"]),
        "latency_ms": int((time.monotonic() - start) * 1000),
    }


def _log_and_score(query: str, run_result: dict, is_agentic: bool, use_judge: bool) -> dict:
    """Optionally scores the answer with the LLM-as-judge, then logs the conversation.
    Returns run_result plus conversation_id/judge, ready for session_state + rendering."""
    judge_verdict = None
    if use_judge:
        try:
            judge_verdict = judge_answer_quality(query, run_result["hits"], run_result["answer"])
        except Exception as exc:
            label = "Agentic" if is_agentic else "Traditional"
            st.warning(f"{label} judge scoring failed (answer above is unaffected): {exc}")

    conversation_id = db.log_conversation(
        question=query, answer=run_result["answer"], method=run_result["db_method"],
        use_rerank=run_result["db_use_rerank"], use_query_rewrite=run_result["db_use_rewrite"],
        filter_dict=run_result["db_filter_dict"], latency_ms=run_result["latency_ms"],
        judge_faithfulness=(judge_verdict or {}).get("faithfulness"),
        judge_context_precision=(judge_verdict or {}).get("context_precision"),
        judge_relevance=(judge_verdict or {}).get("relevance"),
        is_agentic=is_agentic, n_tool_calls=run_result["n_tool_calls"],
    )
    return {**run_result, "conversation_id": conversation_id, "judge": judge_verdict}


def _render_answer_block(container, result: dict, key_prefix: str) -> None:
    """Renders one answer (traditional or agentic) into `container` — either the main
    page body (single mode) or one side of a two-column layout (compare mode)."""
    container.markdown(result["answer"])

    if result.get("tool_calls") is not None:
        container.caption(f"🔍 Agent ran {len(result['tool_calls'])} search(es):")
        for tc in result["tool_calls"]:
            scope = f"sector={tc['sector']}" if tc.get("sector") else (f"ticker={tc['ticker']}" if tc.get("ticker") else "no filter")
            container.caption(f" • \"{tc['query']}\" ({scope})")

    judge = result.get("judge")
    if judge:
        container.caption(f"🤖 LLM-as-judge: {judge.get('reasoning', '')}")
        jcol1, jcol2, jcol3 = container.columns(3)
        faithfulness = judge.get("faithfulness")
        jcol1.metric("Faithfulness", f"{faithfulness:.2f}" if faithfulness is not None else "n/a")
        jcol2.metric("Context Precision", f"{judge.get('context_precision', 0):.2f}")
        jcol3.metric("Relevance", judge.get("relevance", "n/a"))

    feedback_key = f"{key_prefix}_feedback_given"
    if not st.session_state.get(feedback_key):
        container.markdown("**Rate this answer:**")
        comment = container.text_area(
            "Add a comment (optional)",
            key=f"{key_prefix}_comment_{result['conversation_id']}", height=68,
        )
        fcol1, fcol2, _ = container.columns([1, 1, 6])
        if fcol1.button("👍", key=f"{key_prefix}_up"):
            db.log_feedback(result["conversation_id"], is_positive=True, comment=comment)
            st.session_state[feedback_key] = True
            st.session_state[f"{key_prefix}_feedback_positive"] = True
            st.rerun()
        if fcol2.button("👎", key=f"{key_prefix}_down"):
            db.log_feedback(result["conversation_id"], is_positive=False, comment=comment)
            st.session_state[feedback_key] = True
            st.session_state[f"{key_prefix}_feedback_positive"] = False
            st.rerun()
    elif st.session_state.get(f"{key_prefix}_feedback_positive"):
        container.success("👍 Rated helpful. Thanks for the feedback!")
    else:
        container.error("👎 Rated not helpful. Thanks for the feedback!")

    hits = result.get("hits", [])
    container.markdown(f"**Sources ({len(hits)} passages retrieved)**")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        with container.expander(f"[{i}] {m['ticker']} ({m['sector']}) · filing {m['filing_date']}"):
            st.write(h["text"])


with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Mode", ["Traditional pipeline", "Agentic RAG", "Compare both"], index=0,
        help="Traditional: fixed retrieve-then-generate (rag/pipeline.py + rag/generate.py). "
        "Agentic: the LLM decides for itself when and how to search (rag/agent.py) — it can "
        "search once per company/sector, or search again with different wording, adaptively. "
        "Compare both: runs both on the same question and shows them side by side.",
    )
    show_traditional_controls = mode in ("Traditional pipeline", "Compare both")
    show_agentic_controls = mode in ("Agentic RAG", "Compare both")

    if show_traditional_controls:
        sector_label = st.selectbox("Sector filter", ["All", "Tech", "Banks"], index=0)
        sector = {"All": None, "Tech": "tech", "Banks": "banks"}[sector_label]
        ticker_label = st.selectbox("Company filter", ["All"] + sorted(SECTOR), index=0)
        ticker = None if ticker_label == "All" else ticker_label
        k = st.slider("Passages to retrieve (k)", min_value=3, max_value=10, value=6)
        method = st.selectbox(
            "Retrieval method", ["dense", "sparse", "hybrid"], index=0,
            help="dense+rerank is the empirically best config — see eval/results/retrieval_eval.json",
        )
        use_rerank = st.toggle("Cross-encoder reranking", value=True)
        use_rewrite = st.toggle(
            "Query rewriting", value=False,
            help="Helps comparison questions (e.g. 'tech vs banks') by splitting them per side "
            "before retrieval — adds an extra LLM call, so off by default for single-topic questions.",
        )

    if show_agentic_controls:
        max_iterations = st.slider(
            "Max searches", min_value=1, max_value=8, value=MAX_ITERATIONS,
            help="Safety cap on how many searches the agent can run before being forced to answer.",
        )
        st.caption(
            "The agent picks its own search queries and sector/company scoping — "
            "no need to set filters here. Expect more Gemini calls (and higher latency) "
            "than the traditional pipeline, since each search is its own round-trip."
        )

    use_judge = st.toggle(
        "Score this answer (LLM-as-judge)", value=False,
        help="Automatically scores relevance/faithfulness against the retrieved passages, "
        "logged alongside any 👍/👎 feedback. Adds an extra LLM call per question (per mode, "
        "in Compare both) — off by default to avoid extra rate-limit exposure on ordinary use.",
    )
    st.markdown(
        "**Sample questions**\n"
        "- Compare the main AI-related risk factors between tech and banking companies\n"
        "- What are Google's main AI risk factors?\n"
        "- Compare Google vs Microsoft's AI risk factors"
    )

query = st.text_input(
    "Your question",
    placeholder="e.g. Compare the main AI-related risk factors between tech and banking companies",
)
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and query.strip():
    filter_dict = {}
    if show_traditional_controls:
        if sector:
            filter_dict["sector"] = sector
        if ticker:
            filter_dict["ticker"] = ticker
        filter_dict = filter_dict or None

    if mode == "Compare both":
        col1, col2 = st.columns(2)
        col1.markdown("### 🔧 Traditional pipeline")
        col2.markdown("### 🤖 Agentic RAG")
        status1, status2 = col1.empty(), col2.empty()
        status1.info("⏳ Retrieving and generating...")
        status2.info("⏳ Searching and reasoning...")

        with st.spinner("Generating both answers (running in parallel)..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_trad = executor.submit(_run_traditional, query, k, filter_dict, method, use_rerank, use_rewrite)
                future_agentic = executor.submit(_run_agentic, query, max_iterations)
                done_trad = done_agentic = False
                while not (done_trad and done_agentic):
                    if not done_trad and future_trad.done():
                        run_trad = future_trad.result()
                        status1.success(f"✅ Done in {run_trad['latency_ms'] / 1000:.1f}s")
                        done_trad = True
                    if not done_agentic and future_agentic.done():
                        run_agentic = future_agentic.result()
                        status2.success(f"✅ Done in {run_agentic['latency_ms'] / 1000:.1f}s")
                        done_agentic = True
                    if not (done_trad and done_agentic):
                        time.sleep(0.2)

        if use_judge:
            with st.spinner("Scoring both answers..."):
                result_trad = _log_and_score(query, run_trad, is_agentic=False, use_judge=True)
                result_agentic = _log_and_score(query, run_agentic, is_agentic=True, use_judge=True)
        else:
            result_trad = _log_and_score(query, run_trad, is_agentic=False, use_judge=False)
            result_agentic = _log_and_score(query, run_agentic, is_agentic=True, use_judge=False)

        st.session_state["compare_mode"] = True
        st.session_state["last_answer"] = None  # single-mode render is skipped while compare_mode is set
        st.session_state["compare_traditional"] = result_trad
        st.session_state["compare_agentic"] = result_agentic
        st.session_state["cmp_trad_feedback_given"] = False
        st.session_state["cmp_agentic_feedback_given"] = False
    else:
        is_agentic_mode = mode == "Agentic RAG"
        spinner_msg = "Agent is searching and reasoning..." if is_agentic_mode else "Retrieving and generating..."
        with st.spinner(spinner_msg):
            if is_agentic_mode:
                run_result = _run_agentic(query, max_iterations)
            else:
                run_result = _run_traditional(query, k, filter_dict, method, use_rerank, use_rewrite)

        if use_judge:
            with st.spinner("Scoring answer quality..."):
                result = _log_and_score(query, run_result, is_agentic=is_agentic_mode, use_judge=True)
        else:
            result = _log_and_score(query, run_result, is_agentic=is_agentic_mode, use_judge=False)

        st.session_state["compare_mode"] = False
        st.session_state["last_answer"] = result
        st.session_state["single_feedback_given"] = False

if st.session_state.get("compare_mode"):
    st.subheader("Answers")
    col1, col2 = st.columns(2)
    col1.markdown("### 🔧 Traditional pipeline")
    col2.markdown("### 🤖 Agentic RAG")
    _render_answer_block(col1, st.session_state["compare_traditional"], "cmp_trad")
    _render_answer_block(col2, st.session_state["compare_agentic"], "cmp_agentic")
elif st.session_state.get("last_answer"):
    st.subheader("Answer")
    _render_answer_block(st, st.session_state["last_answer"], "single")
elif query == "":
    st.info("Enter a question above, then click **Ask**.")
