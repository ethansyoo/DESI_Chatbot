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
import hashlib
import adql
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

def generate_adql_query(user_input, df_reference, client, temp_val):
    """Generate an ADQL query from natural language using reference CSV data with RL examples."""
    if df_reference is None or df_reference.empty:
        st.error("Reference data is not available. Cannot generate ADQL query.")
        return None

    available_columns = ", ".join(df_reference.columns)
    system_prompt = (
        "You are a helpful assistant that converts natural language queries into ADQL (Astronomical Data Query Language). "
        "Return only the SQL query inside a code block (```sql ... ```) and nothing else. "
        "Avoid explanations, prefaces, or post-processing text. Follow ADQL format strictly.\n\n"
        "Important rules:\n"
        "- ADQL does NOT support the `LIMIT` clause.\n"
        "- Use BETWEEN or JOIN clauses appropriately.\n"
        "- Ensure the query is executable in a TAP service.\n"
    )

    # Build the initial messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Available columns: {available_columns}"}
    ]

    # Incorporate RL-based examples (positive + negative)
    rl_context = adql.find_similar_adql_queries(user_input, adql_collection, top_k=5)

    if rl_context:
        # Positive examples
        if rl_context["positive"]:
            pos_examples = "\n\n".join([
                f"NL: {doc['user_query']}\nADQL:\n{doc['generated_adql']}" 
                for doc in rl_context["positive"]
            ])
            messages.append({
                "role": "system",
                "content": f"Here are good ADQL examples you should follow:\n\n{pos_examples}"
            })

        # Negative examples
        if rl_context["negative"]:
            neg_examples = "\n\n".join([
                f"NL: {doc['user_query']}\nIncorrect ADQL:\n{doc['generated_adql']}" 
                for doc in rl_context["negative"]
            ])
            messages.append({
                "role": "system",
                "content": f"Here are incorrect ADQL examples to avoid:\n\n{neg_examples}"
            })

    # Add conversation history for context
    max_tokens_for_history = 3000  # adjustable
    token_count = 0
    history_messages = []

    adql_history = st.session_state.get("adql_history", [])[::-1]
    for entry in adql_history:
        est_tokens = len(entry["content"]) // 4
        if token_count + est_tokens > max_tokens_for_history:
            break
        history_messages.insert(0, entry)
        token_count += est_tokens

    for msg in history_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add the current query
    messages.append({"role": "user", "content": user_input})

    # Generate ADQL
    try:
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            max_tokens=token_limit,
            temperature=temp_val,
        )
        full_response = response.choices[0].message.content.strip()

        # Extract SQL from ```sql block
        match = re.search(r"```sql\s*(.*?)\s*```", full_response, re.DOTALL)
        adql_query = match.group(1).strip() if match else full_response.strip()
        return adql_query

    except Exception as e:
        st.error(f"Error generating ADQL query: {e}")
        return None

def render_latex_from_response(response_text):
    """
    Enhanced LaTeX rendering that:
    - Converts \[...\], $$...$$ to st.latex blocks
    - Converts \( ... \) and $...$ to inline math
    - Converts (\\alpha) style to \(\\alpha\)
    - Detects LaTeX-y standalone lines and renders them with st.latex
    """
    import re

    # Normalize escaped math environments
    response_text = response_text.replace("\\\\", "\\")
    response_text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", response_text, flags=re.DOTALL)
    response_text = re.sub(r"\\\((.*?)\\\)", r"$\1$", response_text)

    # Convert (\\rho), (\\Lambda), (a(t)), etc. → \(\rho\)
    response_text = re.sub(r"\((\\[a-zA-Z0-9_\{\}\^\.\(\) ]+)\)", r"\\(\1\\)", response_text)

    for line in response_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Block math
        if line.startswith("$$") and line.endswith("$$"):
            st.latex(line.strip("$$"))
            continue

        # Line that looks like full LaTeX
        if re.match(r"^\\?[a-zA-Z\\\{\}_^0-9\s\(\)\+\-\*/=<>|]+$", line) and any(sym in line for sym in ["\\frac", "\\rho", "\\pi", "\\Lambda", "\\dot", "\\ddot", "_", "^", "="]):
            try:
                st.latex(line)
            except:
                st.markdown(line)
            continue

        # Inline LaTeX like \(\rho\)
        st.markdown(line)


# Main application
if "decrypted" not in st.session_state:
    st.title("chatDESI")
    password = st.text_input("Enter your password:", type="password")
    if password:
        try:
            with open("encrypted_credentials.txt", "rb") as cred_file:
                salt = cred_file.readline().strip()
                stored_password_hash = cred_file.readline().strip()
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
            
            # Verify if entered password is correct
            entered_password_hash = hashlib.sha256(password.encode()).hexdigest().encode()
            if entered_password_hash != stored_password_hash:
                st.error("Incorrect password! Please try again.")
                st.stop()

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
    adql_collection = adql.connect_to_adql_collection(
        st.session_state["mongo_username"],
        st.session_state["mongo_password"]
    )

    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = ""
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""
    if "adql_history" not in st.session_state:
        st.session_state["adql_history"] = []
    if "last_adql_query" not in st.session_state:
        st.session_state["last_adql_query"] = ""

    col_left, col_right = st.columns([4, 1])
    with col_right:
        st.write('### Settings')
        mode = st.radio("Select Mode", ["Chat Mode", "ADQL Mode"])
        token_limit = st.number_input(label="Token Limit", min_value=500, max_value=3000, step=100, value=1500)
        temp_val = st.slider(label="Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
        reference_toggle = st.checkbox('Reference Papers', value=True)

        if mode == "ADQL Mode":
            max_records = st.number_input("Set Max Rows (MAXREC)", min_value=100, max_value=50000, step=100, value=500)

    with col_left:
        # st.sidebar.write("## 📄 Upload PDFs to MongoDB")
        # uploaded_pdfs = st.sidebar.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

        # if uploaded_pdfs:
        #     for uploaded_file in uploaded_pdfs:
        #         text = ""
        #         try:
        #             # Extract text from PDF
        #             with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        #                 for page in doc:
        #                     text += page.get_text()

        #             document_hash = pdf.compute_text_hash(text)
        #             existing_document = collection.find_one({"metadata.document_hash": document_hash})

        #             if existing_document:
        #                 st.sidebar.warning(f"Duplicate PDF detected: {uploaded_file.name} (Skipping)")
        #             else:
        #                 pdf.add_pdf_to_db(text, uploaded_file.name, collection)
        #                 st.sidebar.success(f"'{uploaded_file.name}' uploaded and processed successfully!")

        #         except Exception as e:
        #             st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")

        # Load Reference CSV for ADQL
        df_reference = load_reference_data()

        # Mode selection: Chat or ADQL
        # --------------------- CHAT MODE ---------------------
        if mode == "Chat Mode":
            st.write("### chatDESI")

            # User query input
            user_input = st.text_input("Enter your message:", key="chat_input")

            # Send and retry buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                send_query = st.button("Send")
            with col2:
                clear_query = st.button("Clear Chat History")
            with col3:
                retry_query = st.button("Retry")

            # Send user message
            if send_query and user_input:
                st.session_state["last_query"] = user_input
                st.session_state["history"].append({"role": "user", "content": user_input})

                # Retrieve relevant documents
                if reference_toggle:
                    relevant_docs = pdf.find_relevant_docs(user_input, st.session_state['mongo_username'], st.session_state['mongo_password'], top_k=3)
                    st.session_state["relevant_docs"] = relevant_docs  # Store for sidebar display

                # Process user query
                try:
                    context = ''
                    if reference_toggle:
                        context_snippets = "\n\n".join([doc["text"] for doc in relevant_docs[:3]])  # Limit to 1000 chars per doc
                        context = f"Relevant document context:\n\n{context_snippets}"

                    # Add new user query to history
                    st.session_state["history"].append({"role": "user", "content": user_input})

                    # Token-safe chat memory
                    max_history_tokens = 800  # ~adjust depending on temperature + model
                    token_count = len(context) // 4
                    messages = [{"role": "system", "content": context}]

                    # Traverse from latest to earliest
                    chat_history = st.session_state["history"][::-1]
                    history_trimmed = []

                    for entry in chat_history:
                        est_tokens = len(entry["content"]) // 4
                        if token_count + est_tokens > max_history_tokens:
                            break
                        history_trimmed.insert(0, entry)  # maintain original order
                        token_count += est_tokens

                    # Add trimmed memory
                    messages.extend(history_trimmed)
                    messages.insert(0, {
                        "role": "system",
                        "content": (
                            "When including math equations, always wrap them in either `\\[...\\]` for display mode or `\\(...\\)` for inline math. "
                            "Do not write raw LaTeX expressions outside of math environments."
                        )
                    })
                    # Now query the model
                    response = client.chat.completions.create(
                        messages=messages,
                        model="gpt-4o",
                        max_tokens=token_limit,
                        temperature=temp_val,
                    )

                    assistant_message = response.choices[0].message.content
                    st.session_state["history"].append({"role": "assistant", "content": assistant_message})
                    st.session_state['last_response'] = assistant_message
                except Exception as e:
                    st.error(f"Error: {e}")


            # Retry last query
            if retry_query:
                retry_message = f"Previous query: {st.session_state['last_query']}. Retry with improvements. Here was the response: {st.session_state['last_response']}."
                st.session_state["history"].append({"role": "user", "content": retry_message})

                # 🧠 Reuse previous relevant docs (if available)
                reference_context = ""
                if reference_toggle and "relevant_docs" in st.session_state:
                    context_snippets = "\n\n".join([doc["text"] for doc in st.session_state["relevant_docs"][:3]])
                    reference_context = f"Relevant document context:\n\n{context_snippets}"

                try:
                    messages = []
                    if reference_context:
                        messages.append({"role": "system", "content": reference_context})
                    
                    messages.append({"role": "system", "content": "You are a helpful assistant. Improve the previous response, as it is not sufficient."})
                    messages.append({"role": "user", "content": retry_message})

                    response = client.chat.completions.create(
                        messages=messages,
                        model="gpt-4o",
                        max_tokens=token_limit,
                        temperature=temp_val,
                    )
                    assistant_response = response.choices[0].message.content

                    st.session_state["history"].append({"role": "assistant", "content": assistant_response})
                    st.session_state["last_response"] = assistant_response

                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

            # Display last response
            if "last_response" in st.session_state and st.session_state["last_response"]:
                st.write("### chatDESI")
                render_latex_from_response(st.session_state["last_response"])

            # Expandable chat history (instead of sidebar)
            with st.expander("View Full Chat History", expanded=False):
                for chat in st.session_state["history"]:
                    if chat["role"] == "user":
                        st.markdown(f"**User:** {chat['content']}")
                    else:
                        st.code(chat['content'], language="markdown")

            # Sidebar for Relevant Documents

            with st.sidebar:
                st.write("## Relevant Documents")
                if "relevant_docs" in st.session_state and st.session_state["relevant_docs"]:
                    for doc in st.session_state["relevant_docs"]:
                        filename = doc["metadata"].get("filename", "Unnamed")
                        similarity = doc.get("similarity", None)
                        score_str = f" — Similarity: {similarity:.2f}" if similarity is not None else ""

                        text_chunk = doc["text"]

                        with st.sidebar.expander(f"{filename}{score_str}", expanded=False):
                            st.markdown(text_chunk, unsafe_allow_html=True)

                else:
                    st.write("No relevant documents found.")




            # Clear chat history button
            if clear_query:
                st.session_state["history"] = []
                st.session_state["last_query"] = ""
                st.session_state["last_response"] = ""
                st.success("Chat history cleared!")

        # --------------------- ADQL MODE ---------------------
        elif mode == "ADQL Mode":
            st.write("### ADQL Query Builder")

            # Load reference data
            df_reference = load_reference_data()

            if df_reference is not None:
                with st.expander("📖 Reference Data for ADQL Queries", expanded=False):
                    st.dataframe(df_reference)

            # Ensure ADQL session state variables exist
            if "adql_query" not in st.session_state:
                st.session_state["adql_query"] = ""

            if "adql_history" not in st.session_state:
                st.session_state["adql_history"] = []

            if "last_adql_query" not in st.session_state:
                st.session_state["last_adql_query"] = ""

            if "tap_data" not in st.session_state:
                st.session_state["tap_data"] = None

            if "tap_data_updated" not in st.session_state:
                st.session_state["tap_data_updated"] = False

            # User input for natural language ADQL generation
            user_query_nl = st.text_area("Describe your ADQL query in natural language:", height=100, key="adql_nl_input")

            # ADQL Query Box (Users Can Modify It)
            sql_query_input = st.text_area(
                "ADQL Query",
                value=st.session_state["adql_query"],  # Uses the stored value
                height=100,
                key="adql_query_box"  # Assigning a unique key
            )

            # Buttons for Generating and Retrying ADQL Queries
            col1, col2 = st.columns([1, 1])

            with col1:
                generate_query = st.button("Generate ADQL Query")

            with col2:
                retry_query = st.button("Retry Last Query")

            # Handle Generate ADQL Query
            if generate_query:
                if user_query_nl:
                    generated_query = generate_adql_query(user_query_nl, df_reference, client, temp_val)
                    if generated_query:
                        st.session_state["adql_query"] = generated_query  # Overwrite with new query
                        st.session_state["last_adql_query"] = generated_query
                        st.session_state["adql_history"].append({"role": "user", "content": user_query_nl})
                        st.session_state["adql_history"].append({"role": "assistant", "content": generated_query})
                        st.rerun()  # Refresh UI to update text box
                else:
                    st.warning("Please enter a natural language query.")

            # Handle Retry Last Query
            if retry_query:
                if st.session_state["last_adql_query"]:
                    retry_message = f"Retrying last ADQL query: {st.session_state['last_adql_query']}"
                    st.session_state["adql_history"].append({"role": "user", "content": retry_message})

                    try:
                        response = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "Improve the last ADQL query."},
                                {"role": "user", "content": retry_message}
                            ],
                            model="gpt-4o",
                            max_tokens=1500,
                            temperature=0.7,
                        )
                        improved_query = response.choices[0].message.content
                        st.session_state["adql_query"] = improved_query  # Overwrite with improved query
                        st.session_state["adql_history"].append({"role": "assistant", "content": improved_query})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("No previous query to retry.")

            # Add a limit selection UI element (default 500, max 50,000)
            if st.button("Run Query and Graph Data"):
                st.session_state["adql_query"] = sql_query_input  # Store user-edited query before execution
                generated_query = st.session_state["adql_query"]
                if generated_query:
                    tap_service_url = "https://datalab.noirlab.edu/tap/sync"
                    tap_query_url = f"{tap_service_url}?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY={st.session_state['adql_query'].replace(' ', '+')}&MAXREC={max_records}"

                    with st.spinner(f"Fetching up to {max_records} rows from TAP service..."):
                        df = download_tap_data(tap_query_url)

                    if df is not None:
                        entry_id = adql.log_adql_query(
                            adql_collection,
                            user_query_nl,
                            generated_query,
                            execution_success=True,
                            tap_result_rows=len(df)
                        )

                        st.session_state["tap_data"] = df
                        st.session_state["tap_data_updated"] = True
                        st.session_state["last_adql_doc_id"] = entry_id
                        st.session_state["last_adql_user_query"] = user_query_nl
                        st.session_state["last_adql_generated_query"] = generated_query
                        st.session_state["show_feedback_buttons"] = True

                        st.success(f"Data successfully retrieved! Showing up to {max_records} results.")
                        st.write("### TAP Query Result Data:")
                        st.dataframe(df)
                    else:
                        adql.log_adql_query(
                            adql_collection,
                            user_query_nl,
                            st.session_state["adql_query"],
                            execution_success=False,
                            tap_result_rows=0
                        )
                        st.error("Failed to retrieve data. Please check the query or try again.")
                else:
                    st.warning("Please generate or enter an ADQL query first.")

            if (
                st.session_state.get("show_feedback_buttons")
                and st.session_state.get("last_adql_doc_id")
                and st.session_state.get("last_adql_user_query")
                and st.session_state.get("last_adql_generated_query")
            ):
                st.write("#### Was this query helpful?")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("👍 Mark as Helpful"):
                        adql.update_feedback(
                            adql_collection,
                            st.session_state["last_adql_doc_id"],
                            "positive"
                        )
                        st.toast("Thanks for the feedback!")
                        st.session_state["show_feedback_buttons"] = False

                with col2:
                    if st.button("👎 Not Helpful"):
                        adql.update_feedback(
                            adql_collection,
                            st.session_state["last_adql_doc_id"],
                            "negative"
                        )
                        st.toast("Got it — we'll use that to improve.")
                        st.session_state["show_feedback_buttons"] = False

            # ------------------- ADQL HISTORY -------------------
            with st.expander("View ADQL Query History", expanded=False):
                for entry in st.session_state["adql_history"]:
                    if entry["role"] == "user":
                        st.markdown(f"**User:** {entry['content']}")
                    else:
                        st.code(entry["content"], language="sql")

            # Button to Clear ADQL History
            if st.button("Clear ADQL History"):
                st.session_state["adql_history"] = []  # Reset history
                st.success("ADQL history cleared!")  # Show success message
# Feedback footer at the bottom of the screen
feedback_url = "https://forms.gle/pVoAzEgFwKZ4zmXNA"
footer = f"""
    <style>
    .feedback-footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        opacity: 0.6;
        font-size: 14px;
        padding: 5px;
    }}
    </style>
    <div class="feedback-footer">
        Beta testing feedback: <a href="{feedback_url}" target="_blank">Click here</a>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)


