### README.md

# Consulta de Documentos PDF

Este projeto fornece um script Python que processa um documento PDF, divide-o em pequenos pedaços de texto, cria embeddings para esses pedaços, armazena-os em um armazenamento vetorial FAISS e configura um sistema de QA baseado em recuperação usando um modelo LLM local. O script é projetado para ajudar os usuários a consultar e analisar documentos PDF de forma eficiente.

## Recursos

- **Carregamento de PDF**: Carrega e analisa documentos PDF.
- **Divisão de Texto**: Divide o documento em pequenos pedaços de texto gerenciáveis.
- **Criação de Embeddings**: Cria embeddings para os pedaços de texto usando modelos HuggingFace.
- **Armazenamento Vetorial**: Armazena embeddings em um armazenamento vetorial FAISS para recuperação eficiente.
- **QA Baseado em Recuperação**: Configura um sistema de QA baseado em recuperação usando o modelo Ollama do LangChain e modelos de prompt personalizados.

## Requisitos

- Python 3.8+
- `langchain_community`
- `langchain`
- `langchain_core`
- `langchain_text_splitters`
- `langchain_community.llms`
- `langchain_community.embeddings`
- `langchain_community.vectorstores`
- `faiss-cpu` ou `faiss-gpu`
- `huggingface`

## Instalação

1. Clone o repositório:
   ```sh
   git clone https://github.com/seuusuario/interpretador-de-documentos-pdf.git
   cd interpretador-de-documentos-pdf
   ```

2. Instale as dependências necessárias:
   ```sh
   pip install langchain langchain_community langchain_core faiss-cpu huggingface_hub
   ```

## Uso

1. Coloque seu documento PDF no mesmo diretório que o script e nomeie-o como `meu_pdf.pdf`.

2. Execute o script:
   ```
   python pdfquery.py
   ```

## Descrição do Script

- **Carregamento de PDF**: O script usa `PyPDFLoader` para carregar o documento PDF.
- **Divisão de Texto**: Os pedaços de texto são criados usando `CharacterTextSplitter`.
- **Criação de Embeddings**: Embeddings são gerados usando `HuggingFaceEmbeddings`.
- **Armazenamento Vetorial**: Embeddings são armazenados e recuperados de um armazenamento vetorial FAISS.
- **QA Baseado em Recuperação**: Uma cadeia de QA é configurada usando o modelo Ollama do LangChain e modelos de prompt personalizados para responder perguntas com base no conteúdo do documento.

## Personalização

- **Caminho do PDF**: Altere a variável `pdf_path` para apontar para o seu arquivo PDF desejado.
- **Tamanho e Sobreposição dos Pedaços**: Ajuste os parâmetros `chunk_size` e `chunk_overlap` no `CharacterTextSplitter` para atender às suas necessidades.
- **Consulta**: Modifique a variável `query` para especificar diferentes perguntas ou tarefas de análise.

## Licença

Este projeto é licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Agradecimentos

- [LangChain](https://github.com/langchain-ai/langchain)
- [HuggingFace](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)

---
