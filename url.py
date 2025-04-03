import fitz  # PyMuPDF
import re
import csv
import os

def extract_first_reference_url(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    
    for page in doc:
        text += page.get_text()

    doi_pattern = re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b', re.IGNORECASE)
    arxiv_pattern = re.compile(r'arXiv:(\d{4}\.\d{4,5}(v\d+)?)|\bhttps?://arxiv\.org/\S+', re.IGNORECASE)

    matches = []

    for match in doi_pattern.finditer(text):
        matches.append((match.start(), 'doi', match.group(0)))
    for match in arxiv_pattern.finditer(text):
        arxiv_id = match.group(1) if match.group(1) else match.group(0).split("/")[-1]
        matches.append((match.start(), 'arxiv', arxiv_id))

    if not matches:
        return None

    matches.sort(key=lambda x: x[0])
    kind, value = matches[0][1], matches[0][2]

    return f"https://doi.org/{value}" if kind == 'doi' else f"https://arxiv.org/abs/{value}"

def process_pdf_folder_to_csv_recursive(folder_path, output_csv="references.csv"):
    pdf_files = []

    # Recursively walk through all folders and subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    # Write to CSV
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'url'])

        for pdf in pdf_files:
            try:
                url = extract_first_reference_url(pdf)
                writer.writerow([os.path.relpath(pdf, folder_path), url if url else "Not found"])
            except Exception as e:
                writer.writerow([os.path.relpath(pdf, folder_path), f"Error: {str(e)}"])

# Example usage
folder_path = "papers"
process_pdf_folder_to_csv_recursive(folder_path, output_csv="references.csv")
