import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# LLM y función de carga de llaves
def load_LLM(openai_api_key):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm

# Título y cabecera de la página
st.set_page_config(page_title="Pregunte a partir de un archivo CSV con preguntas frecuentes sobre Smash Bros Ultimate")
st.header("Pregunte a partir de un archivo CSV con preguntas frecuentes sobre Smash Bros Ultimate")

st.write("Contacte con [Matias Toro Labra](https://www.linkedin.com/in/luis-matias-toro-labra-b4074121b/) para construir sus proyectos de IA")

# Introducir la clave API de OpenAI
def get_openai_api_key():
    input_text = st.text_input(
        label="OpenAI API Key ",  
        placeholder="Ex: sk-2twmA8tfCb8un4...", 
        key="openai_api_key_input", 
        type="password")
    return input_text

openai_api_key = get_openai_api_key()

if openai_api_key:
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb_file_path = "my_vectordb"

    def create_db():
        loader = CSVLoader(file_path='napoleon-faqs.csv', source_column="prompt")
        documents = loader.load()
        vectordb = FAISS.from_documents(documents, embedding)

        # Save vector database locally
        vectordb.save_local(vectordb_file_path)

    def execute_chain():
        try:
            # Verificar si el archivo de la base de datos existe
            if not os.path.exists(vectordb_file_path):
                st.error(f"No se encontró el archivo de la base de datos de vectores en la ruta: {vectordb_file_path}")
                return None
            
            # Cargar la base de datos vectorial desde la carpeta local
            vectordb = FAISS.load_local(vectordb_file_path, embedding)

            # Crear un recuperador para consultar la base de datos de vectores
            retriever = vectordb.as_retriever(score_threshold=0.7)

            template = """Dado el siguiente contexto y una pregunta, genere una respuesta basada únicamente en este contexto.
            En la respuesta trate de proporcionar la mayor cantidad posible de texto de la sección "respuesta" en el contexto del documento fuente sin hacer muchos cambios.
            Si la respuesta no se encuentra en el contexto, responde "No lo sé". No intentes inventarte una respuesta.

            CONTEXT: {context}

            QUESTION: {question}"""

            prompt = PromptTemplate(
                template=template, 
                input_variables=["context", "question"]
            )
            
            llm = load_LLM(openai_api_key=openai_api_key)

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                input_key="query",
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )

            return chain
        except Exception as e:
            st.error(f"Error al cargar la base de datos de vectores: {e}")
            return None

    if __name__ == "__main__":
        create_db()
        chain = execute_chain()

    btn = st.button("Botón privado: volver a crear la base de datos")
    if btn:
        create_db()

    question = st.text_input("Pregunta: ")

    if question and chain:
        response = chain(question)
        st.header("Respuesta")
        st.write(response["result"])