"""
fixed_size_multi_agent_app.py

Multi-Agent Fixed Width File Processing System
Using LangGraph + Groq LLM

REQUIREMENTS:
- .env file with GROQ_API_KEY=your_key_here
- schema.xlsx file
- data.txt file
"""

import os
import re
import json
import hashlib
import sqlite3
import pandas as pd
from difflib import get_close_matches
from typing import List, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END


# =========================================================
# 🔐 Load Environment Variables
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)


# =========================================================
# 🗄 Database Initialization
# =========================================================

def init_db():
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_name TEXT,
            version INTEGER,
            schema_hash TEXT UNIQUE,
            table_name TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()


init_db()


# =========================================================
# 🧠 TOOLS
# =========================================================

@tool
def read_schema_and_hash_tool(excel_path: str) -> dict:
    df = pd.read_excel(excel_path, header=3)

    df.columns = [c.strip().lower() for c in df.columns]

    expected = {
        "field name": None,
        "start position": None,
        "length": None,
        "req": None
    }

    for key in expected:
        match = get_close_matches(key, df.columns, n=1, cutoff=0.6)
        if match:
            expected[key] = match[0]

    if None in expected.values():
        raise KeyError("Schema columns missing.")

    df = df[list(expected.values())]
    df.columns = ["field_name", "start_position", "length", "required"]
    df = df.dropna(subset=["field_name"])

    schema = [
        {
            "field_name": str(r["field_name"]).strip(),
            "start_position": int(r["start_position"]),
            "length": int(r["length"]),
            "required": str(r["required"]).strip()
        }
        for r in df.to_dict(orient="records")
    ]

    schema_hash = hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode()
    ).hexdigest()

    return {"schema": schema, "schema_hash": schema_hash}


@tool
def check_registry_tool(schema_hash: str) -> dict:
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT table_name, version FROM schema_registry WHERE schema_hash = ?",
        (schema_hash,)
    )

    result = cursor.fetchone()
    conn.close()

    if result:
        return {"exists": True, "table_name": result[0], "version": result[1]}
    return {"exists": False}


@tool
def get_next_version_tool(schema_name: str) -> int:
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT MAX(version) FROM schema_registry WHERE schema_name = ?",
        (schema_name,)
    )

    result = cursor.fetchone()[0]
    conn.close()

    return 1 if result is None else result + 1


@tool
def create_table_tool(table_name: str, columns: list):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    col_defs = []
    for col in columns:
        name = re.sub(r'\W+', '_', col["field_name"]).lower()
        col_defs.append(f'"{name}" TEXT')

    sql = f"""
    CREATE TABLE "{table_name}" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {", ".join(col_defs)}
    );
    """

    cursor.execute(sql)
    conn.commit()
    conn.close()

    return f"Table {table_name} created"


@tool
def register_schema_tool(schema_name: str, version: int,
                         schema_hash: str, table_name: str):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO schema_registry (schema_name, version, schema_hash, table_name)
        VALUES (?, ?, ?, ?)
    """, (schema_name, version, schema_hash, table_name))

    conn.commit()
    conn.close()

    return "Schema registered"


@tool
def parse_fixed_file_tool(file_path: str, schema: list) -> list:
    records = []

    with open(file_path, "r") as f:
        for line in f:
            record = {}
            for col in schema:
                start = col["start_position"] - 1
                end = start + col["length"]
                record[col["field_name"]] = line[start:end].strip()
            records.append(record)

    return records


@tool
def insert_records_tool(table_name: str, records: list) -> int:
    if not records:
        return 0

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cols = records[0].keys()
    col_sql = ", ".join([f'"{c}"' for c in cols])
    placeholders = ", ".join(["?"] * len(cols))

    query = f"INSERT INTO {table_name} ({col_sql}) VALUES ({placeholders})"
    values = [tuple(r[c] for c in cols) for r in records]

    cursor.executemany(query, values)
    conn.commit()
    count = cursor.rowcount
    conn.close()

    return count


# =========================================================
# 🧩 STATE
# =========================================================

class SystemState(TypedDict):
    excel_path: str
    file_path: str
    schema: Optional[List[Dict]]
    schema_hash: Optional[str]
    table_name: Optional[str]
    version: Optional[int]
    rows_inserted: Optional[int]
    messages: List[BaseMessage]


# =========================================================
# 🤖 AGENTS
# =========================================================

schema_agent = create_agent(
    model=llm,
    tools=[read_schema_and_hash_tool],
    system_prompt="Read schema and return schema + schema_hash."
)

db_agent = create_agent(
    model=llm,
    tools=[
        check_registry_tool,
        get_next_version_tool,
        create_table_tool,
        register_schema_tool
    ],
    system_prompt="Handle schema versioning and table creation."
)

file_agent = create_agent(
    model=llm,
    tools=[parse_fixed_file_tool, insert_records_tool],
    system_prompt="Parse fixed file and insert into DB."
)


# =========================================================
# 🔄 GRAPH NODES
# =========================================================

def schema_node(state):
    result = schema_agent.invoke({
        "input": f"Read schema from {state['excel_path']}",
        "messages": state["messages"]
    })

    return {
        "messages": result["messages"],
        "schema": result.get("schema"),
        "schema_hash": result.get("schema_hash")
    }


def db_node(state):
    result = db_agent.invoke({
        "input": f"Handle schema hash {state['schema_hash']}",
        "messages": state["messages"]
    })

    return {
        "messages": result["messages"],
        "table_name": result.get("table_name"),
        "version": result.get("version")
    }


def file_node(state):
    result = file_agent.invoke({
        "input": f"Parse file {state['file_path']}",
        "messages": state["messages"]
    })

    return {
        "messages": result["messages"],
        "rows_inserted": result.get("rows_inserted")
    }


def supervisor(state):
    if not state.get("schema"):
        return {"next": "schema"}
    if not state.get("table_name"):
        return {"next": "db"}
    if not state.get("rows_inserted"):
        return {"next": "file"}
    return {"next": "END"}


# =========================================================
# 🕸 GRAPH
# =========================================================

builder = StateGraph(SystemState)

builder.add_node("supervisor", supervisor)
builder.add_node("schema", schema_node)
builder.add_node("db", db_node)
builder.add_node("file", file_node)

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "schema": "schema",
        "db": "db",
        "file": "file",
        "END": END
    }
)

builder.add_edge("schema", "supervisor")
builder.add_edge("db", "supervisor")
builder.add_edge("file", "supervisor")

app = builder.compile()


# =========================================================
# 🚀 RUN
# =========================================================

if __name__ == "__main__":

    initial_state = {
        "excel_path": "D:\\trinitypro\\schema.xlsx",
        "file_path": "D:\\trinitypro\\data.txt",
        "schema": None,
        "schema_hash": None,
        "table_name": None,
        "version": None,
        "rows_inserted": None,
        "messages": []
    }

    result = app.invoke(initial_state)

    print("\nFINAL RESULT:\n")
    print(result)