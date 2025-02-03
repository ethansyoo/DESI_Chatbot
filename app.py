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

# GitHub raw URL of the reference CSV file
GITHUB_CSV_URL = "https://raw.githubusercontent.com/ethansyoo/DESI_Chatbot/main/columns.csv"

@st.cache_data
def load_reference_data():
    try:
        df = pd.read_csv(
            GITHUB_CSV_URL, 
            sep=",",  # Ensure it's using the correct delimiter
            encoding="utf-8",  # Handle special characters
            on_bad_lines="skip"  # Skip problematic lines instead of failing
        )
        return df
    except Exception as e:
        st.error(f"Error loading reference data: {e}")
        return None


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
        response = requests.get(query_url, timeout=60)  # Fetch data
        response.raise_for_status()  # Ensure request was successful
        if response.text.strip().startswith("<VOTABLE"):
            st.error("TAP service returned an XML response instead of CSV. Check the query format.")
            return None

        # Check if response is empty or contains an error message
        if not response.text.strip():  # Empty response
            st.error("The TAP service returned an empty response. The query might be incorrect.")
            return None
        if "ERROR" in response.text[:100].upper():  # First 100 chars contain "ERROR"
            st.error(f"The TAP service returned an error: {response.text[:300]}")  # Show first 300 chars
            return None

        # Save content as CSV
        with open("tap_query_result.csv", "w", encoding="utf-8") as file:
            file.write(response.text)

        # Try loading into pandas
        df = pd.read_csv("tap_query_result.csv", sep=",", on_bad_lines="skip")
        
        # Validate dataframe structure
        if df.empty:
            st.error("The downloaded CSV is empty. Please check the query.")
            return None
        if len(df.columns) < 1:
            st.error("The CSV file has an unexpected format. Check the TAP service output.")
            return None
        
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Network error while fetching TAP query result: {e}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Failed to parse CSV response: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
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

def generate_adql_query(user_input, df_reference, client, temp_val):
    """Generate an ADQL query from natural language using reference CSV data."""
    if df_reference is None or df_reference.empty:
        st.error("Reference data is not available. Cannot generate ADQL query.")
        return None

    available_columns = ", ".join(df_reference.columns)
    system_prompt = (
        "You are a helpful assistant that converts natural language queries into ADQL (Astronomical Data Query Language). "
        "Return only the SQL query inside a code block (```sql ... ```) and nothing else. Do NOT provide explanations, warnings, or extra text. "
        "Ensure the query is formatted correctly for execution in the TAP service."
        """
        ADQL cheat sheet:

        You cannot use LIMIT in ADQL. It is a rule.

        Example queries: 
        Query: Give me all DESI redshifts within 1 degree of RA=241.050 and DEC=43.45

        Answer: SELECT zpix.z, zpix.zerr, zpix.mean_fiber_ra, zpix.mean_fiber_dec FROM desi_edr.zpix AS zpix WHERE zpix.mean_fiber_ra BETWEEN 240.050 AND 242.050 AND zpix.mean_fiber_dec BETWEEN 42.450 AND 44.450 AND zpix.zwarn = 0

        Query: Give me all DESI redshifts within 1 Mpc of RA=241.050 and DEC=43.45 at redshift z=0.5.

        Answer: SELECT zpix.z, zpix.zerr FROM desi_edr.zpix AS zpix JOIN desi_edr.ztile AS ztile ON zpix.targetid = ztile.targetid WHERE ztile.mean_fiber_ra BETWEEN 240.05 AND 242.05 AND ztile.mean_fiber_dec BETWEEN 42.45 AND 44.45 AND zpix.z BETWEEN 0.495 AND 0.505 AND zpix.zwarn = 0
        """
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Available columns: {available_columns}"},
        {"role": "user", "content": user_input}
    ]
    
    try:
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            max_tokens=token_limit,
            temperature=temp_val,
        )
        full_response = response.choices[0].message.content.strip()

        # Extract SQL query from code block
        match = re.search(r"```sql\s*(.*?)\s*```", full_response, re.DOTALL)
        if match:
            adql_query = match.group(1).strip()  # Extract SQL query inside ```
        else:
            # If no code block is found, assume the whole response is the query
            adql_query = full_response.strip()

        return adql_query
    except Exception as e:
        st.error(f"Error generating ADQL query: {e}")
        return None

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

    # Load Reference CSV for ADQL
    df_reference = load_reference_data()

    # ADQL Checkbox
    use_adql = st.checkbox("Use ADQL Formatting")

    # Show Reference Data if Available
    if df_reference is not None:
        st.write("### Reference Data for ADQL Queries:")
        st.dataframe(df_reference)

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
    # Change query input based on checkbox state
    if use_adql:
        st.write("### ADQL Query Mode Enabled")
        
        # Get user input in natural language
        user_query_nl = st.text_area("Describe your ADQL query in natural language:", height=100)
        
        # Initialize variable for storing generated query
        adql_query = ""

        # Button to trigger query generation
        # Ensure adql_query is stored in session state
        if "adql_query" not in st.session_state:
            st.session_state["adql_query"] = ""

        # Button to generate ADQL Query
        if st.button("Generate ADQL Query"):
            if user_query_nl:
                generated_query = generate_adql_query(user_query_nl, df_reference, client, temp_val)
                if generated_query:
                    st.session_state["adql_query"] = generated_query  # Store in session state
                    st.text_area("Generated ADQL Query:", value=st.session_state["adql_query"], height=100, key="generated_adql_query")
            else:
                st.warning("Please enter a natural language query.")


        # Execute Queries
        # Button to execute query and fetch data
        if st.button("Run Query and Graph Data"):
            if st.session_state["adql_query"]:  # Read from session state
                tap_service_url = "https://datalab.noirlab.edu/tap/sync"
                tap_query_url = f"{tap_service_url}?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY={st.session_state['adql_query'].replace(' ', '+')}"
                print(tap_query_url)
                # Display loading message
                with st.spinner("Fetching data from TAP service..."):
                    df = download_tap_data(tap_query_url)  # Automatically download and parse CSV
                    
                if df is not None:
                    st.session_state["tap_data"] = df
                    st.session_state["tap_data_updated"] = True
                    st.success("Data successfully retrieved!")
                    st.write("### TAP Query Result Data:")
                    st.dataframe(df)  # Display DataFrame preview
                else:
                    st.error("Failed to retrieve data. Please check the query or try again.")
            else:
                st.warning("Please generate an ADQL query first.")


    # Graphing and Visualization

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
