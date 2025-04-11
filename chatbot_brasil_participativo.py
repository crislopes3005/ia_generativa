ique em # Importar as bibliotecas
import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Configurar o título da página
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

# Verificar se o arquivo existe
if os.path.exists("FAQ.docx.pdf"):
    try:
        # Carregar o PDF
        loader = PyPDFLoader("FAQ.docx.pdf")
        docs = loader.load()

        # Dividir o texto em pedaços
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
        chunks = splitter.split_documents(docs)

        # Criar os embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Criar a base vetorial
        vectorstore = FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        st.error(f"Erro ao configurar RAG: {e}")
        vectorstore = None
else:
    st.warning("Arquivo PDF 'FAQ.docx.pdf' não encontrado.")
    vectorstore = None

# Entrada do usuário
user_input = st.text_input("Digite sua dúvida sobre participação social e clique em "Enviar":")

if st.button("Enviar") and user_input.strip():
    relevant_info = None

    # Buscar documentos relevantes
    if vectorstore:
        try:
            docs = vectorstore.similarity_search(user_input, k=2)
            relevant_info = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            st.error(f"Erro ao buscar informações: {e}")

    # Montar o prompt
    full_prompt = system_prompt
    if relevant_info:
        full_prompt += f"\n\nInformações relevantes:\n{relevant_info}"

    try:
        # Gerar a resposta
        response = model.generate_content([f"{st.session_state.conversation}\nUsuário: {user_input}\nAssistente:", full_prompt])
        resposta = response.text
    except Exception as e:
        resposta = f"Erro ao gerar resposta: {e}"

    # Exibir resposta
    st.write(f"**Assistente**: {resposta}")
    st.session_state.conversation += f"\nUsuário: {user_input}\nAssistente: {resposta}"

# Configurar a página
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



