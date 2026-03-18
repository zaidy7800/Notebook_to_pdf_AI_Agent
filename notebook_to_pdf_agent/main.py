from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import nbformat
from nbconvert import WebPDFExporter

load_dotenv()

MODEL_NAME = os.getenv("model", "qwen2.5:7b")
TEMPERATURE = float(os.getenv("temperature", "0.7"))

llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)

@tool
def ipynb_to_pdf(file_path: str) -> str:
    """Converts a Jupyter Notebook (.ipynb) file to a PDF file.
    Input should be the filename or path to the .ipynb file."""
    try:
        if not os.path.exists(file_path):
            return f"Error: The file '{file_path}' was not found."
        with open(file_path, 'r', encoding='utf-8') as f:
            nb_content = nbformat.read(f, as_version=4)
        pdf_exporter = WebPDFExporter()
        pdf_data, _ = pdf_exporter.from_notebook_node(nb_content)
        output_path = file_path.replace(".ipynb", ".pdf")
        with open(output_path, "wb") as f:
            f.write(pdf_data)
        return f"Success! Converted to {output_path}"
    except Exception as e:
        return f"Conversion failed: {str(e)}"

agent = create_react_agent(
    model=llm,
    tools=[ipynb_to_pdf],
    prompt=SystemMessage(content="You are a Helpful AI Agent")
)

file_p = input("Enter File Path: ").strip()

response = agent.invoke(
    {"messages": [{"role": "user", "content": f"Convert this ipynb file {file_p} into pdf"}]}
)

print(response["messages"][-1].content)