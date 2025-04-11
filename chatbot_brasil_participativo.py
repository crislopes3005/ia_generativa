import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ✅ TEM QUE SER A PRIMEIRA CHAMADA DO STREAMLIT
st.set_page_config(page_title="Chatbot - Brasil Participativo", page_icon="🤖")

# Configurar API da Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

system_prompt = """
Você é um assistente virtual especializado em participação social. Ao receber uma mensagem, sua tarefa é:
1. Fornecer respostas claras e objetivas sobre participação social e sobre a plataforma Brasil Participativo.
2. Recomendar a participação social e, caso solicitado, sugira processos participativos abertos na plataforma Brasil Participativo.
3. Não comente sobre o documento fornecido.
4. Utilize informações do documento PDF carregado para enriquecer suas respostas quando relevante.
5. Seja educado!
"""

@st.cache_resource
def setup_rag():
    try:
        if not os.path.exists("FAQ.docx.pdf"):
            st.warning("Arquivo PDF 'FAQ.docx.pdf' não encontrado.")
            return None
        
        loader = PyPDFLoader("FAQ.docx.pdf")
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)
        return db
    except Exception as e:
        st.error(f"Erro ao configurar RAG: {e}")
        return None

vectorstore = setup_rag()

def get_relevant_info(query):
    if not vectorstore:
        return None
    try:
        docs = vectorstore.similarity_search(query, k=2)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        st.error(f"Erro ao buscar informações: {e}")
        return None

def generate_response(user_input, history):
    relevant_info = get_relevant_info(user_input)
    full_prompt = system_prompt
    if relevant_info:
        full_prompt += f"\n\nInformações relevantes:\n{relevant_info}"
    
    try:
        response = model.generate_content([f"{history}\nUsuário: {user_input}\nAssistente:", full_prompt])
        return response.text
    except Exception as e:
        return f"Erro ao gerar resposta: {e}"


st.title("🤖 Chatbot - Plataforma Brasil Participativo")

if "history" not in st.session_state:
    st.session_state.history = system_prompt
if "conversation" not in st.session_state:
    st.session_state.conversation = ""

user_input = st.text_input("Digite sua dúvida sobre participação social:")
if st.button("Enviar") and user_input.strip():
    st.session_state.conversation += f"\nUsuário: {user_input}"
    response = generate_response(user_input, st.session_state.conversation)
    st.write(f"**Assistente**: {response}")
    st.session_state.conversation += f"\nAssistente: {response}"



