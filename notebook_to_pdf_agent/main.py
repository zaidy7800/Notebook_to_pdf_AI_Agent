import os
import nbformat
from nbconvert import WebPDFExporterimport os
import nbformat
from nbconvert import WebPDFExporter
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Initialize LLM (Ollama)
# -----------------------------
llm = ChatOllama(
    model=os.getenv("model", "qwen2.5:7b"),
    temperature=float(os.getenv("temperature", 0.7))
)

# -----------------------------
# TOOL: Convert Notebook → PDF
# -----------------------------
@tool
def convert_notebook_to_pdf(file_path: str) -> str:
    """
    Converts a Jupyter notebook file (.ipynb) into a PDF.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"

    if not file_path.endswith(".ipynb"):
        return "Error: File must be a .ipynb file"

    try:
        # Read notebook
        with open(file_path, "r", encoding="utf-8") as f:
            nb_content = nbformat.read(f, as_version=4)

        # Export to PDF
        pdf_exporter = WebPDFExporter()
        pdf_data, _ = pdf_exporter.from_notebook_node(nb_content)

        # Save PDF
        output_path = file_path.replace(".ipynb", ".pdf")
        with open(output_path, "wb") as f:
            f.write(pdf_data)

        return f"PDF successfully created at: {output_path}"

    except Exception as e:
        return f"Conversion failed: {str(e)}"


# -----------------------------
# Bind tool to LLM (LangChain Agent)
# -----------------------------
llm_with_tools = llm.bind(
    tools=[{
        "name": "convert_notebook_to_pdf",
        "description": "Converts a Jupyter notebook file (.ipynb) into a PDF",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the .ipynb file"
                }
            },
            "required": ["file_path"]
        }
    }]
)

# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    print("Notebook-to-PDF AI Agent Ready!")
    print("Type natural requests like:")
    print("- 'Convert question.ipynb to PDF'")
    print("- 'Please convert my notebook sample.ipynb'")
    print("Type 'exit' to quit.\n")

    system_message = """You are an AI assistant. When user asks to convert a notebook to PDF, call the convert_notebook_to_pdf tool.

Example: If user says "Convert question.ipynb to PDF", use tool: convert_notebook_to_pdf with file_path="question.ipynb"

IMPORTANT: Use the tool when needed. Just call it and wait for the result."""

    while True:
        user_input = input("Enter your request: ").strip()
        if user_input.lower() == "exit":
            print("Exiting agent...")
            break

        try:
            # Create messages
            messages = [
                ("system", system_message),
                ("user", user_input)
            ]

            # Invoke with tools - this IS a LangChain agent
            response = llm_with_tools.invoke(messages)

            # Check for tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_args = tool_call.get('args', {})

                    if 'file_path' in tool_args and tool_args['file_path']:
                        result = convert_notebook_to_pdf.invoke(tool_args['file_path'])
                        print("\nAgent Response:", result, "\n")
            else:
                print("\nAgent Response:", response.content, "\n")

        except Exception as e:
            print("Error:", str(e))

def ipynb_to_pdf(file_path: str) -> str:
    """Converts a Jupyter Notebook (.ipynb) file to PDF."""
    try:
        if not os.path.exists(file_path):
            return f"Error: The file '{file_path}' was not found."
        
        # Read the notebook
        with open(file_path, 'r', encoding='utf-8') as f:
            nb_content = nbformat.read(f, as_version=4)
        
        # Export to PDF
        pdf_exporter = WebPDFExporter()
        pdf_data, _ = pdf_exporter.from_notebook_node(nb_content)
        
        # Save PDF
        output_path = file_path.replace(".ipynb", ".pdf")
        with open(output_path, "wb") as f:
            f.write(pdf_data)
        
        return f"Success! Converted to {output_path}"
    
    except Exception as e:
        return f"Conversion failed: {str(e)}"

if __name__ == "__main__":
    file_path = input("Enter the path to your .ipynb file: ").strip()
    result = ipynb_to_pdf(file_path)
    print(result)
