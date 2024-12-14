import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd

# Initialize Pinecone
try:
    pinecone_client = Pinecone(
        api_key="pcsk_6CfDiG_8BC8mKNf18uLCHnh73xa1hTi6Ag8Fbz7GszRgUHcTBc5f1fZXuDNn1hq7gJLvf9"  # Replace with your Pinecone API key
    )
    st.success("Pinecone initialized successfully!")
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {e}")
    st.stop()

# Define Pinecone index name
index_name = "symptom-disease-index-osb-final3"

# Handle index creation or access
try:
    pinecone_client.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    st.success(f"Index '{index_name}' created successfully!")
except Exception as e:
    if "ALREADY_EXISTS" in str(e):
        st.info(f"Index '{index_name}' already exists. Proceeding to use the existing index.")
    else:
        st.error(f"An error occurred while handling the index: {e}")
        st.stop()

# Access the Pinecone index
try:
    index = pinecone_client.Index(index_name)
except Exception as e:
    st.error(f"Failed to access the Pinecone index: {e}")
    st.stop()

# Load SentenceTransformer model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.success("Embedding model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load embedding model: {e}")
    st.stop()

# Load the RAG model for conversational responses
try:
    rag_model = pipeline("text2text-generation", model="google/flan-t5-base")
    st.success("RAG model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load RAG model: {e}")
    st.stop()

# Load disease dataset
disease_dataset_path = "Symptom2Disease.csv"  # Replace with your dataset path
try:
    diseases_data = pd.read_csv(disease_dataset_path)
    all_diseases = diseases_data['label'].unique().tolist()
    st.success("Disease dataset loaded successfully!")
except Exception as e:
    st.error(f"Failed to load disease dataset: {e}")
    st.stop()

# Load precautions from file
precautions = {}
try:
    with open("precautions.txt", "r") as f:
        lines = f.readlines()
        current_disease = None
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            if current_disease is None:  # Start of a new disease section
                current_disease = line
                precautions[current_disease] = []
            elif i + 1 < len(lines) and lines[i + 1].strip():
                precautions[current_disease].append(line)
            else:
                precautions[current_disease].append(line)
                current_disease = None  # Reset for the next disease
    st.success("Precautions loaded successfully!")
except Exception as e:
    st.error(f"Failed to load precautions file: {e}")
    st.stop()

# Initialize session state for conversation
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# Chat Interface
st.title("Disease Symptom Chatbot")

# Display conversation history
st.subheader("Chat")
for message in st.session_state["conversation_history"]:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "bot":
        st.markdown(f"**Bot:** {message['content']}")

# Dynamic text box rendering
with st.form("chat_form", clear_on_submit=True):
    temp_user_input = st.text_input("Type your question here:", placeholder="E.g., What could cause a fever and sore throat?")
    submitted = st.form_submit_button("Send")

if submitted:
    if temp_user_input.strip():
        try:
            # Check if query is about precautions
            if "precaution" in temp_user_input.lower():
                # Extract the disease from the query (assume it's after "precaution for <disease>")
                disease_name = temp_user_input.split("for")[-1].strip().capitalize()
                if disease_name in precautions:
                    bot_response = f"The precautions for {disease_name} are:\n\n- " + "\n- ".join(precautions[disease_name])
                else:
                    bot_response = f"Sorry, I couldn't find precautions for {disease_name}."
            else:
                # Append user query to conversation history
                st.session_state["conversation_history"].append({"role": "user", "content": temp_user_input})

                # Generate embedding for the user query
                input_embedding = embedding_model.encode(temp_user_input).tolist()

                if len(input_embedding) != 384:  # Adjust based on your embedding model's dimensions
                    st.error("Unexpected embedding dimensions. Check model configuration.")
                else:
                    # Query Pinecone for similar symptoms
                    results = index.query(vector=input_embedding, top_k=5, include_metadata=True)

                    # Select the closest match
                    if results.matches:
                        best_match = results.matches[0]  # Closest match is the first result
                        closest_disease = best_match.metadata.get('label', 'Unknown Disease')
                        closest_score = best_match.score

                        # Generate a conversational response
                        context = "\n".join(
                            [f"{msg['role']}: {msg['content']}" for msg in st.session_state["conversation_history"]]
                        )
                        bot_response = (
                            f"The most likely disease is: {closest_disease} (confidence: {closest_score:.2f}).\n"
                        )
                        bot_response += rag_model(context, max_length=256)[0]['generated_text']
                    else:
                        bot_response = "I couldn't find any relevant diseases for your symptoms."

            # Append bot response to conversation history
            st.session_state["conversation_history"].append({"role": "bot", "content": bot_response})

            # Display the updated chat interface
            st.experimental_rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query before proceeding.")