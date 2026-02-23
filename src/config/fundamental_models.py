from langchain_ollama import OllamaEmbeddings, ChatOllama

embeddings = OllamaEmbeddings(
     model="nomic-embed-text",
 )

llm_qwen3 = ChatOllama(
    model="qwen3:1.7b",
    temperature=0.5
)

llm_qwen3_m = ChatOllama(
    model="qwen3:8b",
    temperature=0.5
)
