# utils.py - NEW FREE VERSION

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from pypdf import PdfReader
from langchain.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import os
import uuid  # Still need uuid for session management


# --- LLM and EMBEDDINGS SETUP ---

# 1. Define the FREE open-source Embedding Model
# This model is loaded locally via SentenceTransformerEmbeddings
def create_embeddings_load_data():
    # Using the same model as your original code, which is free and loads from HF
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# 2. Define the FREE Summarization LLM
# We use a fast, small, free-to-use summarization model from Hugging Face Hub.
# NOTE: This requires the HUGGINGFACEHUB_API_TOKEN environment variable, which is FREE to get.
def get_llm():
    # Use a small, free summarization model. T5 is a common choice.
    # Replace 't5-small' with another if needed, but T5 works well for summarization.
    try:
        llm = HuggingFaceHub(
            repo_id="t5-small",
            model_kwargs={"temperature": 0.1, "max_length": 100}
        )
        return llm
    except Exception as e:
        print(f"Error loading HuggingFaceHub LLM: {e}")
        # Fallback for local testing if API key is not set
        return None

    # --- PDF PROCESSING ---


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def create_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name, "unique_id": unique_id},
        ))
    return docs


# --- CHROMA DB (FREE LOCAL VECTOR STORE) FUNCTIONS ---

# Use a global dictionary to store vectorstores in memory during the session
VECTOR_STORES = {}


def push_to_chroma(unique_id, embeddings, docs):
    # Chroma is an in-memory vector store here, associated with the unique_id
    global VECTOR_STORES
    vectorstore = Chroma.from_documents(docs, embeddings)
    VECTOR_STORES[unique_id] = vectorstore
    return vectorstore


def pull_from_chroma(unique_id):
    global VECTOR_STORES
    return VECTOR_STORES.get(unique_id)


def similar_docs(query, k, unique_id):
    vectorstore = pull_from_chroma(unique_id)
    if not vectorstore:
        raise ValueError("Vector store not initialized for this session.")

    # Use 'similarity_search_with_score' which Chroma supports
    similar_docs_with_score = vectorstore.similarity_search_with_score(query, k=int(k))

    # We need to map this output to match the expected structure of your original code
    # The output format is (Document, score), which is what your old code used.
    return similar_docs_with_score


# --- SUMMARIZATION ---

def get_summary(current_doc):
    llm = get_llm()
    if llm is None:
        return "Summarization service is unavailable. Please check HUGGINGFACEHUB_API_TOKEN."

    # Using a simple prompt for the T5-small model
    prompt_template = """Write a concise summary of the following text, focusing on technical skills and experience:
    "{text}"
    CONCISE SUMMARY:"""

    # Only map_reduce chain type is efficient for long documents with small LLMs
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",  # 'stuff' is faster for smaller documents, 'map_reduce' is safer for very long ones.
        prompt=PromptTemplate(template=prompt_template, input_variables=["text"])
    )

    # Summarize the single document
    summary = chain.run([current_doc])
    return summary