import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from openai import OpenAI
import pdf
import fitz
import re
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def decrypt_data(encrypted_data, key):
    nonce = encrypted_data[:12]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, encrypted_data[12:], None).decode()


def clear_text():
    st.session_state['input'] = ''


def clear_history():
    st.session_state["history"] = []
    st.session_state["last_query"] = ''
    st.session_state["last_response"] = ''


def parse_and_render_content(label, content):
    st.markdown(f"**{label}:**")
    parts = re.split(r'```', content)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part.strip())
        else:
            st.code(part.strip(), language='python')


def download_tap_data(query_url):
    try:
        response = requests.get(query_url)
        response.raise_for_status()
        with open("tap_query_result.txt", "w") as file:
            file.write(response.text)
        df = pd.read_csv("tap_query_result.txt", sep=",")
        return df
    except Exception as e:
        st.error(f"Failed to download or parse TAP query result: {e}")
        return None


def find_relevant_docs(query, client, collection, top_k=3):
    query_embedding = pdf.create_embedding(query, client)
    if not query_embedding:
        return []

    query_embedding = np.array(query_embedding).reshape(1, -1)
    documents = list(collection.find())
    embeddings = np.array([doc["embedding"] for doc in documents if "embedding" in doc])

    if not embeddings.size:
        return []

    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    sorted_indices = similarities.argsort()[::-1][:top_k]
    relevant_docs = [documents[i] for i in sorted_indices]
    return relevant_docs


# Main application
if "decrypted" not in st.session_state:
    st.title("chatDESI")
    password = st.text_input("Enter your password:", type="password")
    if password:
        try:
            with open("encrypted_credentials.txt", "rb") as cred_file:
                salt = cred_file.readline().strip()
                encrypted_openai_api_key = cred_file.readline().strip()
                encrypted_mongo_username = cred_file.readline().strip()
                encrypted_mongo_password = cred_file.readline().strip()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = kdf.derive(password.encode())
            openai_api_key = decrypt_data(encrypted_openai_api_key, key)
            mongo_username = decrypt_data(encrypted_mongo_username, key)
            mongo_password = decrypt_data(encrypted_mongo_password, key)
            st.session_state["decrypted"] = True
            st.session_state["openai_api_key"] = openai_api_key
            st.session_state["mongo_username"] = mongo_username
            st.session_state["mongo_password"] = mongo_password
            st.success("Credentials decrypted successfully!")
        except:
            st.error("Invalid password or corrupted credentials file.")
            st.stop()
else:
    client = OpenAI(api_key=st.session_state["openai_api_key"])
    collection = pdf.connect_to_mongo(st.session_state["mongo_username"], st.session_state["mongo_password"])
    st.title("DESI Chatbot with Query and Graphing Features")
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = ""
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""

    # ADQL Checkbox
    use_adql = st.checkbox("Use ADQL Formatting")

    # PDF Upload
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            text = ""
            with fitz.open("pdf", uploaded_file.read()) as doc:
                for page in doc:
                    text += page.get_text()
            document_hash = pdf.compute_text_hash(text)
            existing_document = collection.find_one({"metadata.document_hash": document_hash})
            if not existing_document:
                pdf.add_pdf_to_db(text, uploaded_file.name, client, collection)
                st.success(f"'{uploaded_file.name}' uploaded and processed successfully.")

    # TAP Query Section
    tap_query_url = st.text_input("Enter TAP Query URL:")
    if st.button("Run TAP Query and Graph Data"):
        if tap_query_url:
            st.write("Downloading and processing TAP query result...")
            df = download_tap_data(tap_query_url)
            if df is not None:
                st.session_state["tap_data"] = df
                st.session_state["tap_data_updated"] = True
            else:
                st.warning("No data retrieved. Check your TAP query URL.")
        else:
            st.warning("Please provide a valid TAP query URL.")

    if "tap_data" in st.session_state:
        df = st.session_state["tap_data"]

        if st.session_state.get("tap_data_updated", False):
            st.write("**TAP Query Result Data:**")
            st.dataframe(df)
            st.session_state["tap_data_updated"] = False

        st.write("### Graphing Options")
        graph_type = st.selectbox("Select Graph Type", ["Line", "Bar", "Scatter", "Heatmap"], key="graph_type")

        if graph_type == "Heatmap":
            x_column = st.selectbox("Select X-axis column", df.columns, key="heatmap_x_column")
            y_column = st.selectbox("Select Y-axis column", df.columns, key="heatmap_y_column")
            agg_column = st.selectbox("Select Aggregation Column", df.columns, key="heatmap_agg_column")

            if st.button("Generate Heatmap"):
                try:
                    heatmap_data = df.pivot_table(index=y_column, columns=x_column, values=agg_column, aggfunc="mean", fill_value=0)
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
                    plt.title(f"Heatmap of {agg_column} with {y_column} vs {x_column}")
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Error generating heatmap: {e}")
        else:
            x_column = st.selectbox("Select X-axis column", df.columns, key="x_column")
            y_column = st.selectbox("Select Y-axis column", df.columns, key="y_column")

            if st.button("Generate Graph"):
                plt.figure(figsize=(10, 6))
                if graph_type == "Line":
                    plt.plot(df[x_column], df[y_column], label=f"{y_column} vs {x_column}")
                elif graph_type == "Bar":
                    plt.bar(df[x_column], df[y_column], label=f"{y_column} vs {x_column}")
                elif graph_type == "Scatter":
                    plt.scatter(df[x_column], df[y_column], label=f"{y_column} vs {x_column}", alpha=0.6)
                plt.title(f"{graph_type} Graph: {y_column} vs {x_column}")
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.legend()
                st.pyplot(plt)

    # User Query Section
    user_input = st.text_input("Query:", key="input")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        send_query = st.button("Send")
    with col2:
        st.button("Clear text box", on_click=clear_text)
    with col3:
        st.button("Clear chat history", on_click=clear_history)
    with col4:
        retry_query = st.button("Retry")

    token_limit = st.number_input(label="Token Limit", min_value=500, max_value=5000, step=100, value=1500)
    temp_val = st.slider(label="Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

    if send_query and user_input:
        st.session_state["last_query"] = user_input
        st.session_state["history"].append({"role": "user", "content": user_input})

        relevant_docs = find_relevant_docs(user_input, client, collection, top_k=3)
        context_snippets = []

        if relevant_docs:
            st.write("### Relevant Documents:")
            for doc in relevant_docs:
                snippet = doc['text'][:500]  # Truncate snippet to avoid excess length
                context_snippets.append(f"Filename: {doc['metadata']['filename']}\nSnippet: {snippet}")
                st.write(f"**Filename:** {doc['metadata']['filename']}")
                st.write(f"**Snippet:** {snippet}...")
        else:
            st.write("No relevant documents found.")

        # Combine user input and relevant document context
        system_prompt = (
            "You are a helpful assistant. Below are snippets from relevant documents "
            "related to the user's query. Use this information to provide a comprehensive response."
        )
        document_context = "\n\n".join(context_snippets)
        user_message = user_input

        # Build the messages list for OpenAI API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Relevant document context:\n\n{document_context}"},
            {"role": "user", "content": user_message}
        ]

        try:
            response = client.chat.completions.create(
                messages=messages,
                model="chatgpt-4o-latest",
                max_tokens=token_limit,
                temperature=temp_val,
            )
            assistant_message = response.choices[0].message.content
            st.session_state["history"].append({"role": "assistant", "content": assistant_message})
            st.session_state['last_response'] = assistant_message
        except Exception as e:
            st.error(f"Error: {e}")


    if retry_query:
        retry_message = f"Previous query: {st.session_state['last_query']}. Retrying with improvements."
        st.session_state["history"].append({"role": "user", "content": retry_message})
        st.write("Retrying the last query...")

    for chat in st.session_state["history"]:
        if chat["role"] == "user":
            parse_and_render_content("USER", chat['content'])
        else:
            parse_and_render_content("chatDESI", chat['content'])
