from langchain_ollama import OllamaEmbeddings, ChatOllama

embeddings = OllamaEmbeddings(
     model="nomic-embed-text",
 )

llm_qwen3 = ChatOllama(
    model="qwen3:1.7b",
    temperature=0.5
)
