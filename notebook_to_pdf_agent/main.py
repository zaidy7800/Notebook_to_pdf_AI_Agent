import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import sqlite3

# Notebook tools
import nbformat
from nbconvert import WebPDFExporter

# LangChain
from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_tool_call,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
)
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool

# LangGraph
from langgraph.checkpoint.memory import MemorySaver

# Ollama Model
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "minimax-m2.5:cloud")
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.7"))
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "notebook_agent.db")

# -----------------------------
# TOOL: Convert Notebook to PDF
# -----------------------------
@tool
def notebook_to_pdf(notebook_path: str) -> str:
    """Convert a Jupyter Notebook (.ipynb) file into a PDF."""
    try:
        if not os.path.exists(notebook_path):
            return f"Error: File not found at {notebook_path}"

        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        exporter = WebPDFExporter()
        body, resources = exporter.from_notebook_node(nb)

        output_path = notebook_path.replace('.ipynb', '.pdf')
        with open(output_path, 'wb') as f:
            f.write(body)

        return f"PDF successfully created at: {output_path}"

    except Exception as e:
        return f"Conversion failed: {str(e)}"

# -----------------------------
# Middleware
# -----------------------------
@wrap_tool_call
def handle_tool_call_error(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return f"Tool error: {str(e)}"

tool_retry = ToolRetryMiddleware(
    max_retries=2,
    tools=["notebook_to_pdf"],
    on_failure="continue",
)

model_retry = ModelRetryMiddleware(
    max_retries=2,
    on_failure="continue",
)

# -----------------------------
# System Prompt
# -----------------------------
custom_system_prompt = f"""
You are a Notebook to PDF Converter Assistant.

Your ONLY job is:
- Take a notebook path from the user
- Use the notebook_to_pdf tool to convert it
- Return the result clearly

Rules:
- Always use the tool when a file path is given
- Do NOT explain extra theory
- Be concise

Today: {datetime.today()}
"""

# -----------------------------
# Agent Setup
# -----------------------------
def run_agent():
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=MODEL_TEMP
    )

    memory = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=[notebook_to_pdf],
        system_prompt=custom_system_prompt,
        middleware=[handle_tool_call_error, tool_retry, model_retry],
        checkpointer=memory,
        name="Notebook PDF Agent"
    )

    return agent

# -----------------------------
# Streaming Response
# -----------------------------
def stream_response(agent, query: str, config: dict):
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode="values"
    ):
        latest_message = chunk["messages"][-1]

        if isinstance(latest_message, AIMessage):
            if isinstance(latest_message.content, str) and latest_message.content:
                print(f"Agent: {latest_message.content}")

# -----------------------------
# CLI
# -----------------------------
def main():
    print("\n=== Notebook to PDF Converter Agent ===")
    print("Type path of .ipynb file or 'exit' to quit")

    agent = run_agent()
    config = {"configurable": {"thread_id": "notebook_thread"}}

    while True:
        try:
            query = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            sys.exit(0)

        try:
            stream_response(agent, query, config)
        except Exception as err:
            print(f"Error: {err}")


if __name__ == "__main__":
    main()
