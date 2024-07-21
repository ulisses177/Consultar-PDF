import os
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from PIL import Image
import io
import base64
import gradio as gr

class DocumentWrapper:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

def generate_hash(urls):
    combined_string = ''.join(sorted(urls))
    return hashlib.md5(combined_string.encode()).hexdigest()[:8]

def scrape_web_pages(urls, storage_folder):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    documents = []
    image_paths = []
    for url in urls:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(separator='\n').strip()
                if text:
                    documents.append({"content": text, "metadata": {"source": url}})
                    print(f"Scraped content from {url}")
                else:
                    print(f"No text content found in {url}")
                images = soup.find_all('img')
                for idx, img in enumerate(images):
                    img_url = img.get('src')
                    if img_url:
                        if not img_url.startswith('http'):
                            img_url = url + img_url
                        img_response = requests.get(img_url)
                        if img_response.status_code == 200:
                            img_data = img_response.content
                            try:
                                img = Image.open(io.BytesIO(img_data))
                                img.verify()  # Check if image is valid
                                img_path = os.path.join(storage_folder, f'web_image_{len(image_paths)+1}.png')
                                with open(img_path, 'wb') as img_file:
                                    img_file.write(img_data)
                                image_paths.append((url, img_path))
                            except (IOError, SyntaxError) as e:
                                print(f"Skipping unsupported image from {img_url}: {e}")
            else:
                print(f"Failed to retrieve {url}: Status code {response.status_code}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        time.sleep(1)  # Add delay to avoid getting blocked
    return documents, image_paths

def describe_images_with_llava(image_paths, llava, documents):
    descriptions = {}
    for url, image_path in image_paths:
        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        try:
            llm_with_image_context = llava.bind(images=[image_b64])
            image_description = llm_with_image_context.invoke(
    "Descreva os elementos da imagen sucintamente. Inclua informações sobre elementos de interface, objetos visíveis, ignore setas e elementos visuais para chamar a atenção do usuário, apenas forneça uma descrição do que esses elementos visuais estão querendo salientar. Formato da resposta: [listagem dos elementos visuais da tela]. Exemplos: [Tabela indicando um calendário]; [Interface com botões para criar ou cancelar]"
)

            context = find_image_context(url, documents)
            if context:
                image_description = f"{context}\n\n{image_description}"
            if url not in descriptions:
                descriptions[url] = []
            descriptions[url].append(image_description)
        except ValueError as e:
            print(f"Skipping image description for {image_path}: {e}")
    return descriptions

def find_image_context(url, documents):
    for doc in documents:
        if doc['metadata']['source'] == url:
            return doc['content']
    return ""

def split_documents(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size=512, chunk_overlap=64, separator="\n"
    )
    docs = []
    for document in documents:
        content = document["content"]
        metadata = document["metadata"]
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            docs.append(DocumentWrapper(chunk, metadata))
    return docs

def process_web_content(urls):
    short_hash = generate_hash(urls)
    storage_folder = os.path.join("data", short_hash)
    index_path = os.path.join(storage_folder, "faiss_index_web")
    embedder = HuggingFaceEmbeddings()

    if os.path.exists(index_path):
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(
            index_path, embedder, allow_dangerous_deserialization=True
        )
    else:
        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)

        print("Scraping web pages...")
        documents, image_paths = scrape_web_pages(urls, storage_folder)
        
        if not documents:
            raise ValueError("No documents were scraped from the provided URLs.")
        
        print(f"Scraped {len(documents)} documents.")
        
        llava = Ollama(model="llava")
        descriptions = describe_images_with_llava(image_paths, llava, documents)

        print("Splitting documents...")
        docs = split_documents(documents)
        
        if not docs:
            raise ValueError("No documents were split into chunks.")
        
        print(f"Split documents into {len(docs)} chunks.")
        
        for doc in docs:
            url = doc.metadata.get("source", None)
            if url and url in descriptions:
                for desc in descriptions[url]:
                    doc.page_content += f"\n\n{desc}"

        vectorstore = FAISS.from_documents(docs, embedder)
        vectorstore.save_local(index_path)
        print("Created and saved new vectorstore.")

    prompt_de_cadeia_de_retorno = PromptTemplate.from_template(
        "Responda a quaisquer perguntas de uso com base exclusivamente no contexto abaixo:"
        "<contexto>"
        "{context}"
        "</contexto>"
    )
    combine_docs_chain = create_stuff_documents_chain(
        Ollama(model="qwenPT"), prompt_de_cadeia_de_retorno
    )
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(), combine_docs_chain
    )

    return retrieval_chain

def query_web_pages(query, urls):
    retrieval_chain = process_web_content(urls)
    res = retrieval_chain.invoke({"input": query})
    return res['answer']

def summarize_answer(query, answer):
    summarizer = Ollama(model="qwenPT")
    summary_prompt = f"Resuma sua resposta, o mais sussintamente possível, em relação a essa pergunta: '{query}'\n\nResposta: {answer}"
    summary = summarizer.invoke(summary_prompt)
    return summary

def handle_query(query, urls):
    answer = query_web_pages(query, urls)
    summary = summarize_answer(query, answer)
    print(answer)
    return f"Resposta: {summary}"

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Web Page Query Interface")
        query_input = gr.Textbox(lines=2, placeholder="Enter your query here...")
        urls_input = gr.Textbox(lines=5, placeholder="Enter URLs separated by commas...")
        output = gr.Textbox(lines=10, placeholder="Response will be shown here...")
        query_button = gr.Button("Query")

        def handle_click(query, urls):
            urls_list = [url.strip() for url in urls.split(',')]
            return handle_query(query, urls_list)
        
        query_button.click(handle_click, inputs=[query_input, urls_input], outputs=output)

    demo.launch()

if __name__ == "__main__":
    gradio_interface()
