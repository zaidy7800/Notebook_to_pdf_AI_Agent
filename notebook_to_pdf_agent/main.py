import os
import nbformat
from nbconvert import WebPDFExporter

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
