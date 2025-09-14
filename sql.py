# text_to_sql_langgraph.py
"""
A minimal LangGraph-based Natural Language ➜ SQL agent.

Features
--------
1. Accepts a NL question, e.g. "How many employees live in Mumbai?".
2. Uses an LLM to translate the question into SQL.
3. Executes the SQL against a SQLite database.
4. Returns a conversational answer plus the raw result rows.

The graph (DAG) has three nodes:
    ┌────────────────────┐      ┌───────────────────┐      ┌───────────────────┐
    │  generate_sql      │────► │  execute_sql      │────► │  formulate_answer │
    └────────────────────┘      └───────────────────┘      └───────────────────┘

All state lives in a TypedDict so it's easy to extend (add `schema` or `error`
keys, etc.)

This is *not* production‑ready (no error‑handling loops, no SQL validation),
but it's a working starting point you can iterate on.
"""

from __future__ import annotations

import os
import sqlite3
from typing import List, TypedDict, Any, Dict

from langchain.chat_models import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_groq import ChatGroq
from langchain.sql_database import SQLDatabase
from langgraph.graph import StateGraph

# ---------------------------------------------------------------------------
# 0. Local SQLite setup (demo DB) -------------------------------------------
# ---------------------------------------------------------------------------

db_path = "demo.db"
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Simple demo schema
    cur.execute(
        """
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            city TEXT,
            salary INTEGER
        );
        """
    )
    cur.executemany(
        "INSERT INTO employees (name, city, salary) VALUES (?, ?, ?);",
        [
            ("Alice", "Mumbai", 90000),
            ("Bob", "Delhi", 85000),
            ("Charlie", "Mumbai", 95000),
        ],
    )
    conn.commit()
    conn.close()

sql_db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# ---------------------------------------------------------------------------
# 1. State schema -----------------------------------------------------------
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Graph state passed between nodes."""

    messages: List[BaseMessage]
    sql: str | None
    result: Any | None


S = AgentState  # LangGraph helper

# ---------------------------------------------------------------------------
# 2. LLM setup --------------------------------------------------------------
# ---------------------------------------------------------------------------

llm = ChatGroq(model="llama-3.1-8b-instant")

# Prompt template for NL → SQL
SYSTEM_PROMPT = (
    "You are a senior data engineer. Convert the user's question into a "
    "valid SQL query for the given SQLite database. Only output SQL; do not "
    "explain."
)

# ---------------------------------------------------------------------------
# 3. Node definitions -------------------------------------------------------
# ---------------------------------------------------------------------------

def generate_sql(state: AgentState) -> Dict[str, Any]:
    """Node 1: Use LLM to translate NL question to SQL."""
    question = state["messages"][-1].content
    prompt = f"{SYSTEM_PROMPT}\n\nDatabase schema:\n{sql_db.get_table_info()}\n\nQuestion: {question}"
    sql_text = llm.invoke(prompt).content.strip()
    return {"sql": sql_text}


def execute_sql(state: AgentState) -> Dict[str, Any]:
    """Node 2: Run SQL and capture rows."""
    sql_query = state["sql"]
    try:
        rows = sql_db.run(sql_query)
        return {"result": rows}
    except Exception as e:
        # In production, send to an error‑handling node
        return {"result": f"SQL execution failed: {e}"}


def formulate_answer(state: AgentState) -> Dict[str, Any]:
    """Node 3: Turn the raw rows into a conversational answer."""
    rows = state["result"]
    question = state["messages"][-1].content
    answer_prompt = (
        "You are a helpful analyst. Given the user's question and the SQL "
        "result rows, craft a concise answer.\n\nQuestion: "
        f"{question}\nRows: {rows}"
    )
    response = llm.invoke(answer_prompt).content
    return {"messages": [AIMessage(content=response)]}


# ---------------------------------------------------------------------------
# 4. Build the graph --------------------------------------------------------
# ---------------------------------------------------------------------------

builder = StateGraph(state=S)

builder.add_node("generate_sql", generate_sql)
builder.add_node("execute_sql", execute_sql)
builder.add_node("answer", formulate_answer)

builder.add_edge("__start__", "generate_sql")
builder.add_edge("generate_sql", "execute_sql")
builder.add_edge("execute_sql", "answer")

builder.set_entry_point("generate_sql")

graph = builder.compile()

# ---------------------------------------------------------------------------
# 5. Example invocation -----------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_q = "How many employees live in Mumbai?"
    init_state: AgentState = {"messages": [HumanMessage(content=user_q)], "sql": None, "result": None}

    final = graph.invoke(init_state)
    # The final state merges all updates; messages now includes the AI reply.
    print("AI answer:\n", final["messages"][-1].content)
    print("Generated SQL:\n", final["sql"])
    print("Raw rows:\n", final["result"])
