from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from pypdf import PdfReader
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import uuid

API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
hf_client = InferenceClient(token=API_TOKEN)
# --- LLM and EMBEDDINGS SETUP ---

def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


def get_llm():
    try:
        llm = HuggingFaceHub(
            repo_id="google/t5-small-lm-adapt",
            model_kwargs={"temperature": 0.1, "max_length": 200}
        )
        return llm
    except Exception as e:
        print(f"Error loading HuggingFaceHub LLM: {e}")
        return None


# --- PDF PROCESSING ---
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
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
VECTOR_STORES = {}


def push_to_chroma(unique_id, embeddings, docs):
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
    similar_docs_with_score = vectorstore.similarity_search_with_score(query, k=int(k))
    return similar_docs_with_score


# --- SUMMARIZATION ---
def get_summary(doc):
    try:
        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
        result = hf_client.summarization(text)
        return result.summary_text if hasattr(result, "summary_text") else str(result)
    except Exception as e:
        return f"Summarization service error: {e}"
