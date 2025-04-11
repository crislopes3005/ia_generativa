# Importar bibliotecas
import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ✅ Primeira chamada obrigatória do Streamlit
st.set_page_config(page_title="Chatbot - Brasil Participativo", page_icon="🤖")

# Título
st.title("🤖 Chatbot - Plataforma Brasil Participativo")

# Inicializa histórico da conversa
if "conversation" not in st.session_state:
    st.session_state.conversation = """
Você é um assistente virtual especializado em participação social. Ao receber uma mensagem, sua tarefa é:
1. Fornecer respostas claras e objetivas sobre participação social e sobre a plataforma Brasil Participativo.
2. Recomendar a participação social e, caso solicitado, sugira processos participativos abertos na plataforma Brasil Participativo.
3. Não comente sobre o documento fornecido.
4. Utilize informações do documento PDF carregado para enriquecer suas respostas quando relevante.
5. Seja educado!
"""

# Configurar a API da Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# RAG: carregar e processar o PDF
vectorstore = None
if os.path.exists("FAQ.pdf"):
    try:
        loader = PyPDFLoader("FAQ.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"Erro ao configurar RAG: {e}")
else:
    st.warning("Arquivo PDF 'FAQ.pdf' não encontrado.")

# Interface de entrada
user_input = st.text_input("Digite sua dúvida sobre participação social e clique em Enviar:")

if st.button("Enviar") and user_input.strip():
    relevant_info = ""

    if vectorstore:
        try:
            docs = vectorstore.similarity_search(user_input, k=2)
            relevant_info = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            st.error(f"Erro ao buscar informações: {e}")

    full_prompt = st.session_state.conversation
    if relevant_info:
        full_prompt += f"\n\nInformações relevantes:\n{relevant_info}"

    try:
        response = model.generate_content([f"{st.session_state.conversation}\nUsuário: {user_input}\nAssistente:", full_prompt])
        resposta = response.text
    except Exception as e:
        resposta = f"Erro ao gerar resposta: {e}"

    # Mostrar resposta
    st.write(f"**Assistente**: {resposta}")
    st.session_state.conversation += f"\nUsuário: {user_input}\nAssistente: {resposta}"

    st.session_state.conversation += f"\nAssistente: {response}"



