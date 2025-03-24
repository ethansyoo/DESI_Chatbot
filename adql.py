import datetime
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def find_similar_adql_queries(query_text, collection, top_k=3):
    query_embedding = embed_text(query_text)
    query_vector = np.array(query_embedding).reshape(1, -1)

    cursor = list(collection.find({"embedding": {"$exists": True}, "execution_success": True}))
    if not cursor:
        return []

    all_embeddings = np.array([doc["embedding"] for doc in cursor])
    similarities = cosine_similarity(query_vector, all_embeddings).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return [cursor[i] for i in top_indices]
