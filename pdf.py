from pymongo import MongoClient
import fitz  # PyMuPDF for PDF text extraction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import os
import torch
from transformers import AutoTokenizer, AutoModel
import re

# Load Hugging Face Model for Tokenization & Embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Change this to a 1536/3072-d model if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# MongoDB Connection
def connect_to_mongo(mongo_username, mongo_password):
    client = MongoClient(f"mongodb+srv://{mongo_username}:{mongo_password}@cluster89780.vxuht.mongodb.net/?appName=mongosh+2.3.3&tls=true")
    db = client["pdf_database"]
    collection = db["pdf_documents"]
    return collection

# Hash Function for Deduplication
def compute_text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Overlapping Sliding Window Chunking (150 Tokens per Chunk, 50 Token Overlap)
def sliding_window_chunks(text, tokenizer, chunk_size=150, overlap=50):
    tokens = tokenizer.tokenize(text)  # Convert text to tokens
    chunks = []
    step = chunk_size - overlap  # Sliding step

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)  # Convert back to text
        chunks.append(chunk_text)

    return chunks

# Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Embedding Function Using Hugging Face Model
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0).numpy()  # Convert to NumPy array for MongoDB storage

# Add Processed PDF Text & Chunks to MongoDB
def add_pdf_to_db(text, filename, collection):
    document_hash = compute_text_hash(text)
    existing_document = collection.find_one({"metadata.document_hash": document_hash})
    
    if existing_document:
        print(f"Duplicate document detected: '{filename}' - Skipping entire document.")
        return

    text_chunks = sliding_window_chunks(text, tokenizer)

    for i, chunk in enumerate(text_chunks):
        embedding = embed_text(chunk).tolist()
        document = {
            "text": chunk,
            "embedding": embedding,
            "metadata": {
                "filename": filename,
                "document_hash": document_hash,
                "chunk_index": i
            }
        }
        collection.insert_one(document)
    print(f"Document '{filename}' added successfully with {len(text_chunks)} chunks.")

# Process PDFs and Store in MongoDB
def load_pdfs_into_db(pdf_dir, mongo_username, mongo_password):
    collection = connect_to_mongo(mongo_username, mongo_password)
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            add_pdf_to_db(text, filename, collection)

def find_relevant_docs(query, mongo_username, mongo_password, top_k=3):
    collection = connect_to_mongo(mongo_username, mongo_password)
    query_embedding = embed_text(query)

    # Fetch all stored embeddings from MongoDB
    documents = list(collection.find())
    embeddings = np.array([doc["embedding"] for doc in documents if "embedding" in doc])

    if embeddings.size == 0:
        return []

    # Compute Cosine Similarities
    query_embedding = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    sorted_indices = similarities.argsort()[::-1][:top_k]

    # Add similarity scores to docs
    relevant_docs = []
    for i in sorted_indices:
        doc = documents[i]
        doc["similarity"] = round(float(similarities[i]), 3)  # Add similarity score
        relevant_docs.append(doc)

    return relevant_docs

def highlight_keywords(text, query):
    terms = re.findall(r'\b\w+\b', query.lower())
    for term in set(terms):
        pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
        # Highlight with HTML span
        text = pattern.sub(r'<span style="background-color: #ffff00; font-weight: bold;">\1</span>', text)
    return text



# Clear MongoDB Collection
def clear_collections(mongo_username, mongo_password):
    collection = connect_to_mongo(mongo_username, mongo_password)
    collection.delete_many({})
