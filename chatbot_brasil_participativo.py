import streamlit as st
import sys
from importlib import util

# Verificação das dependências
def check_package(package_name):
    if util.find_spec(package_name) is None:
        st.error(f"Pacote {package_name} não encontrado. Instale com: pip install {package_name}")
        st.stop()

check_package("sentence_transformers")
check_package("transformers")
check_package("torch")

import streamlit as st
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Verificação rigorosa das dependências
try:
    from sentence_transformers import SentenceTransformer
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
except ImportError as e:
    st.error(f"ERRO CRÍTICO: Falha na importação de dependências. Execute:\n\n"
             f"pip uninstall -y numpy transformers sentence-transformers torch\n"
             f"pip install numpy==1.23.5 torch==2.0.1 transformers==4.30.2 sentence-transformers==2.2.2")
    st.stop()

# Configurar a chave da API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Criar o modelo
model = genai.GenerativeModel("gemini-1.5-flash")

system_prompt = """
Você é um assistente virtual especializado em participação social. Ao receber uma mensagem, sua tarefa é:
1. Fornecer respostas claras e objetivas sobre participação social e sobre a plataforma Brasil Participativo.
2. Recomendar a participação social e, caso solicitado, sugira processos participativos abertos na plataforma Brasil Participativo.
3. Não comente sobre o documento fornecido.
4. Utilizar informações do documento PDF carregado para enriquecer suas respostas quando relevante.
5. Seja educado!
"""

def setup_rag():
    try:
        if not os.path.exists("FAQ.docx.pdf"):
            return None
        
        pdf_loader = PyPDFLoader("FAQ.docx.pdf")
        documents = pdf_loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
        texts = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="modelo_local",
            model_kwargs={"local_files_only": True}
        )
        db = FAISS.from_documents(texts, embeddings)
        
        return db
    except Exception as e:
        st.error(f"Erro ao configurar RAG: {e}")
        return None

vectorstore = setup_rag()

def get_relevant_info(query):
    if vectorstore is None:
        return None
    
    try:
        docs = vectorstore.similarity_search(query, k=2)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        st.error(f"Erro na busca de informações: {e}")
        return None

def generate_response(user_input, history):
    relevant_info = get_relevant_info(user_input)
    
    prompt = system_prompt
    if relevant_info:
        prompt += f"\n\nInformações relevantes do documento:\n{relevant_info}"
    
    try:
        response = model.generate_content([f"{history}\nUsuário: {user_input}\nAssistente:", prompt])
        return response.text
    except Exception as e:
        return f"Erro ao gerar resposta: {e}"

# Interface
st.set_page_config(page_title="Chatbot - Plataforma Brasil Participativo", page_icon="🤖")
st.title("🤖Chatbot - Plataforma Brasil Participativo 🤝")

# Histórico de conversa
if "history" not in st.session_state:
    st.session_state.history = system_prompt
if "conversation" not in st.session_state:
    st.session_state.conversation = ""

# Chat
user_input = st.text_input("Digite sua dúvida sobre participação social e sobre a plataforma Brasil Participativo:")
if st.button("Enviar") and user_input.strip():
    st.session_state.conversation += f"\nUsuário: {user_input}"
    response = generate_response(user_input, st.session_state.conversation)
    st.write(f"**Assistente**: {response}")
    st.session_state.conversation += f"\nAssistente: {response}"
