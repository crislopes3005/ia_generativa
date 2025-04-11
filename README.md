# 🤖 Chatbot com Retrieval-Augmented Generation (RAG) para a Plataforma Brasil Participativo

## 📌 Sobre o Projeto

Este projeto foi desenvolvido por **Cristiane Lopes de Assis** como parte de uma avaliação de aprendizagem. Ele tem como objetivo ampliar as capacidades do chatbot da plataforma **Brasil Participativo**, permitindo que usuários tirem dúvidas de forma mais eficiente sobre a participação social, a própria plataforma e os processos participativos do Governo Federal.

Atualmente, o chatbot da plataforma possui respostas limitadas e pré-definidas. Este projeto propõe um novo modelo com **RAG (Retrieval-Augmented Generation)**, proporcionando respostas mais precisas, dinâmicas e contextualizadas.

---

## 🧠 Tecnologias Utilizadas

- **[Gemini 1.5 Flash](https://ai.google.dev/gemini)** – Modelo pré-treinado de linguagem da Google
- **[Streamlit](https://streamlit.io/)** – Framework para criação de interfaces web interativas
- **[LangChain](https://www.langchain.com/)** – Framework para aplicações de linguagem com RAG
- **[Sentence Transformers](https://www.sbert.net/)** – Para criação de embeddings e busca semântica
- **PyPDFLoader** – Para carregamento e processamento de arquivos PDF

---

## 🧰 Metodologia

- Utilização de uma base de dados em PDF contendo:
  - As principais dúvidas dos usuários sobre a plataforma
  - Detalhes de todos os processos participativos cadastrados
- Integração do modelo **Gemini 1.5 Flash** com um pipeline de RAG
- Construção da interface com **Streamlit**
- Uso de RAG para recuperar trechos relevantes e gerar respostas com base nos documentos

---

## ▶️ Como Usar

1. Execute o app localmente ou acesse via [Streamlit Cloud](https://streamlit.io/cloud) (se aplicável)
2. Escreva sua dúvida na caixa de entrada:
   ```
   Digite aqui a sua dúvida sobre a plataforma Brasil Participativo e seus processos participativos
   ```
3. O chatbot responderá com base nos documentos disponíveis e no modelo de linguagem integrado.

---

## ✅ Resultados Esperados

Com a nova solução de chatbot:

- Usuários conseguem sanar dúvidas de forma mais precisa e eficiente
- Aumento do engajamento na plataforma e nos processos participativos
- Contribuição para a melhoria contínua das políticas públicas por meio da ampliação da participação social
