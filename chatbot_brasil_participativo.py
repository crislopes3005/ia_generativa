import streamlit as st

# Configurações básicas
st.set_page_config(page_title="Teste Chatbot", page_icon="🤖")
st.title("🚀 Teste de Funcionamento do App")

st.write("✅ O app carregou com sucesso!")
st.write("Essa é uma versão mínima apenas para testar se o ambiente Streamlit Cloud está funcionando corretamente.")

# Input simples
pergunta = st.text_input("Digite algo para testar:")
if st.button("Enviar"):
    st.success(f"Você digitou: {pergunta}")



