import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_core.prompts import PromptTemplate

def main():
    # Caminho para o arquivo PDF
    pdf_path = "telegestão.pdf"
    print("Carregando PDF e criando chunks...")

    # Carregue o PDF e divida-o em documentos individuais
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    print(documents)
    
    # Divide os documentos em partes para incorporação
    text_splitter = CharacterTextSplitter(
        chunk_size=512, chunk_overlap=64, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    for x in docs:
        print(x)
        print("----------------")
    
    print("Criando vetor de embeddings...")



    # Cria os embeddings para os chunks do documento
    embedder = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedder)

    # Salve o armazenamento de vetores localmente
    vectorstore.save_local("faiss_index_react")

    print("Carregando vector store...")

    # Carregue o armazenamento de vetores para recuperação
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embedder, allow_dangerous_deserialization=True
    )

    print("Configurando uma corrente de QA...")

    # Configure a cadeia de controle de qualidade de recuperação usando um prompt predefinido do hub
    # Traduzindo de langchain-ai/retrieval-qa-chat
    #retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    prompt_de_cadeia_de_retorno=PromptTemplate.from_template(
    "Responda a quaisquer perguntas de uso com base exclusivamente no contexto abaixo:"
    "<contexto>"
    "{context}"
    "</contexto>"
)
    combine_docs_chain = create_stuff_documents_chain(
        Ollama(model="qwenPT"), prompt_de_cadeia_de_retorno
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    # Defina a consulta
    query = "analize este documento"
    print(f"Querying: {query}")

    # Invoca a cadeia de recuperação com a consulta e imprima a resposta
    res = retrieval_chain.invoke({"input": query})
    print(f"Answer: {res['answer']}")

if __name__ == "__main__":
    main()
