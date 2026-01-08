
import os
import logging
from typing import List, Optional
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class IBGEKnowledgeBase:
    """Base de conhecimento sobre a API do IBGE para auxiliar o coletor."""

    def __init__(
            self,
            embeddings_model: str = "nomic-embed-text",
            connection_string: Optional[str] = None,
            collection_name: str = "ibge_docs"
    ):
        """
        Inicializa a base de conhecimento.

        Args:
            embeddings_model: Modelo Ollama para embeddings
            connection_string: String de conexão PostgreSQL (usa env var se não fornecido)
            collection_name: Nome da coleção no PGVector
        """
        self.embeddings = OllamaEmbeddings(model=embeddings_model)

        # Usa connection string fornecida ou busca de variável de ambiente
        if connection_string is None:
            connection_string = os.getenv(
                "PGVECTOR_CONNECTION",
                "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
            )

        self.connection_string = connection_string
        self.collection_name = collection_name

        # Inicializa o vector store
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True,
        )

        logger.info(f"PGVector inicializado - Coleção: {self.collection_name}")

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Busca na base de conhecimento.

        Args:
            query: Consulta de busca
            k: Número de resultados a retornar

        Returns:
            Lista de documentos relevantes
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"Busca por '{query}' retornou {len(results)} resultados")
            return results
        except Exception as e:
            logger.error(f"Erro ao buscar no vector store: {e}")
            return []

    def search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Busca na base de conhecimento com scores de similaridade.

        Args:
            query: Consulta de busca
            k: Número de resultados a retornar

        Returns:
            Lista de tuplas (documento, score)
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.debug(f"Busca com score por '{query}' retornou {len(results)} resultados")
            return results
        except Exception as e:
            logger.error(f"Erro ao buscar com score: {e}")
            return []

    def get_api_examples(self, endpoint_type: str = None) -> List[str]:
        """
        Obtém exemplos de uso da API do IBGE.

        Args:
            endpoint_type: Tipo de endpoint (opcional)

        Returns:
            Lista de exemplos
        """
        query = "exemplo de uso API REST endpoint"
        if endpoint_type:
            query = f"{endpoint_type} {query}"

        results = self.search(query, k=3)
        examples = []

        for doc in results:
            content = doc.page_content
            lines = content.split('\n')

            for line in lines:
                if any(keyword in line.lower() for keyword in
                       ['curl', 'http://', 'https://', 'api/v', 'exemplo:', 'exemplo de']):
                    examples.append(line.strip())

        return examples[:5]

    def get_table_info(self, table_code: str = None) -> List[Document]:
        """
        Obtém informações sobre tabelas do SIDRA.

        Args:
            table_code: Código da tabela (opcional)

        Returns:
            Documentos com informações da tabela
        """
        query = f"tabela SIDRA {table_code}" if table_code else "tabelas SIDRA IBGE"
        return self.search(query, k=5)

    def get_variable_info(self, variable_name: str) -> List[Document]:
        """
        Obtém informações sobre variáveis específicas.

        Args:
            variable_name: Nome da variável

        Returns:
            Documentos com informações da variável
        """
        return self.search(f"variável {variable_name} IBGE", k=5)

    def check_collection_exists(self) -> bool:
        """
        Verifica se a coleção já existe e tem documentos.

        Returns:
            True se a coleção existe e tem documentos
        """
        try:
            # Tenta fazer uma busca simples
            results = self.vector_store.similarity_search("test", k=1)
            return len(results) > 0
        except Exception as e:
            logger.debug(f"Coleção ainda não populada: {e}")
            return False


# Singleton para uso global
_ibge_knowledge_base: Optional[IBGEKnowledgeBase] = None


def get_ibge_knowledge_base(
        embeddings_model: str = "nomic-embed-text",
        connection_string: Optional[str] = None,
        collection_name: str = "ibge_docs"
) -> IBGEKnowledgeBase:
    """
    Obtém ou cria a instância singleton da base de conhecimento.

    Args:
        embeddings_model: Modelo de embeddings
        connection_string: String de conexão PostgreSQL
        collection_name: Nome da coleção

    Returns:
        Instância da base de conhecimento
    """
    global _ibge_knowledge_base

    if _ibge_knowledge_base is None:
        _ibge_knowledge_base = IBGEKnowledgeBase(
            embeddings_model=embeddings_model,
            connection_string=connection_string,
            collection_name=collection_name
        )
        logger.info("Instância singleton da base de conhecimento criada")

    return _ibge_knowledge_base


def reset_knowledge_base():
    """Reseta a instância singleton (útil para testes)."""
    global _ibge_knowledge_base
    _ibge_knowledge_base = None
    logger.info("Base de conhecimento resetada")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testando instância do PGVector...")
    kb = get_ibge_knowledge_base()

    if kb.check_collection_exists():
        print("✓ Coleção existe e contém documentos")

        # Teste de busca
        results = kb.search("API IBGE", k=3)
        print(f"✓ Teste de busca retornou {len(results)} resultados")
    else:
        print("⚠ Coleção vazia - execute o script de carga de documentos")