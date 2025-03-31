import datetime
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0).numpy()

def connect_to_adql_collection(mongo_username, mongo_password, db_name="pdf_database", collection_name="adql_feedback"):
    client = MongoClient(f"mongodb+srv://{mongo_username}:{mongo_password}@cluster89780.vxuht.mongodb.net/?appName=mongosh+2.3.3&tls=true")
    db = client[db_name]
    collection = db[collection_name]
    return collection

def log_adql_query(collection, user_query, generated_adql, execution_success, tap_result_rows):
    embedding = embed_text(user_query).tolist()
    entry = {
        "user_query": user_query,
        "generated_adql": generated_adql,
        "timestamp": datetime.datetime.utcnow(),
        "execution_success": execution_success,
        "tap_result_rows": tap_result_rows,
        "user_feedback": None,
        "retry_count": 0,
        "embedding": embedding
    }
    result = collection.insert_one(entry)
    return result.inserted_id

def update_feedback(collection, entry_id, feedback, retry_count=0):
    result = collection.update_one(
        {"_id": entry_id},
        {"$set": {
            "user_feedback": feedback,
            "retry_count": retry_count
        }}
    )
    return result.modified_count

def get_successful_queries(collection, limit=5):
    return list(collection.find({
        "execution_success": True,
        "user_feedback": "positive"
    }).sort("timestamp", -1).limit(limit))

def find_similar_adql_queries(query_text, collection, top_k=5, include_negative=True):
    query_embedding = embed_text(query_text)
    query_vector = np.array(query_embedding).reshape(1, -1)

    # Fetch only documents with embeddings and feedback
    all_docs = list(collection.find({
        "embedding": {"$exists": True},
        "execution_success": True,
        "user_feedback": {"$in": ["positive", "negative"]}  # Only examples with labeled feedback
    }))

    if not all_docs:
        return {"positive": [], "negative": []}

    # Compute similarity scores
    embeddings = np.array([doc["embedding"] for doc in all_docs])
    similarities = cosine_similarity(query_vector, embeddings).flatten()

    # Attach similarity scores
    for i, doc in enumerate(all_docs):
        doc["similarity"] = similarities[i]

    # Split by feedback
    positive = sorted(
        [doc for doc in all_docs if doc["user_feedback"] == "positive"],
        key=lambda x: x["similarity"],
        reverse=True
    )[:top_k]

    negative = []
    if include_negative:
        negative = sorted(
            [doc for doc in all_docs if doc["user_feedback"] == "negative"],
            key=lambda x: x["similarity"],
            reverse=True
        )[:top_k]

    return {"positive": positive, "negative": negative}

import streamlit as st

def render_openai_with_math(text):
    """
    Detects LaTeX-like equations and renders them nicely in Streamlit.
    Converts raw inline or block LaTeX (e.g. k^2\psi = ...) into rendered form.
    """

    # Convert things like [ ... ] or bare equations into $$ ... $$ blocks
    def convert_blocks(txt):
        # Convert \[...\] or bare square brackets into LaTeX blocks
        txt = re.sub(r'\[\s*(\\?.*?)\s*\]', r'$$\1$$', txt)
        return txt

    text = convert_blocks(text)

    # Split by double newlines or blocks
    paragraphs = re.split(r"\n{2,}", text)

    for p in paragraphs:
        if "$$" in p:
            # For block equations
            matches = re.findall(r"\$\$(.*?)\$\$", p, flags=re.DOTALL)
            for match in matches:
                st.latex(match.strip())
        else:
            # Inline equations are supported inside markdown
            st.markdown(p, unsafe_allow_html=True)
