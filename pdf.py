from pymongo import MongoClient
import fitz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

def connect_to_mongo(mongo_username, mongo_password):
    client = MongoClient(f"mongodb+srv://{mongo_username}:{mongo_password}@cluster89780.vxuht.mongodb.net/?appName=mongosh+2.3.3&tls=true")
    db = client["pdf_database"]
    collection = db["pdf_documents"]
    return collection

def compute_text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def split_text_into_chunks(text, max_tokens=3000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        current_token_count += len(word) // 4
        if current_token_count >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_token_count = len(word) // 4
        current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def add_pdf_to_db(text, filename, client, collection):
    document_hash = compute_text_hash(text)
    existing_document = collection.find_one({"metadata.document_hash": document_hash})
    
    if existing_document:
        print(f"Duplicate document detected: '{filename}' - Skipping entire document.")
        return

    text_chunks = split_text_into_chunks(text)

    for i, chunk in enumerate(text_chunks):
        embedding = create_embedding(chunk, client)
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
    print(f"Document '{filename}' added successfully.")


def load_pdfs_into_db(pdf_dir, mongo_username, mongo_password, client):
    collection = connect_to_mongo(mongo_username, mongo_password)
    import os
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            add_pdf_to_db(text, filename, client, collection)
            
def create_embedding(text_chunk, client):
    response = client.embeddings.create(input=text_chunk, model="text-embedding-3-large")
    embedding = response.data[0].embedding
    return embedding

def load_pdfs_into_index(client, pdf_dir, mongo_username, mongo_password):
    collection = connect_to_mongo(mongo_username, mongo_password)
    import os
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            add_pdf_to_db(text, filename, client, collection)

def find_relevant_docs(query, mongo_username, mongo_password, client, top_k=3):
    collection = connect_to_mongo(mongo_username, mongo_password)
    query_embedding = create_embedding(query, client)
    
    if query_embedding is None or len(query_embedding) == 0:
        return []

    query_embedding = np.array(query_embedding).reshape(1, -1)

    documents = list(collection.find())
    embeddings = np.array([doc["embedding"] for doc in documents if "embedding" in doc and doc["embedding"]])

    if embeddings.size == 0:
        return []

    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    sorted_indices = similarities.argsort()[::-1][:top_k]
    relevant_docs = [documents[i] for i in sorted_indices]
    return relevant_docs
