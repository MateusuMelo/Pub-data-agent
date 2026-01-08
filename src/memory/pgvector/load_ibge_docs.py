import os
import sys
import logging
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.knowledge.vector_store import get_ibge_knowledge_base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 🔥 EMBEDDING MODEL (OLLAMA)
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)


def load_identificadores_csv(csv_file_path: str) -> List[Document]:
    """
    Carrega identificadores IBGE (assunto, tema, agregado, etc)
    e transforma cada linha em um documento semanticamente controlado.
    """

    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_file_path}")

    logger.info(f"Carregando identificadores IBGE: {csv_file_path}")

    df = pd.read_csv(csv_file_path)

    required_columns = {"tipo", "id", "nome"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV deve conter as colunas: {required_columns}"
        )

    documents: List[Document] = []

    for _, row in df.iterrows():
        tipo = str(row["tipo"]).strip()
        ident_id = str(row["id"]).strip()
        nome = str(row["nome"]).strip()

        if not tipo or not ident_id or not nome:
            logger.warning("Linha ignorada por dados incompletos")
            continue

        # 🧠 TEXTO SEMÂNTICO FORTE E DIRECIONADO
        page_content = (
            f"IBGE OFFICIAL STATISTICAL IDENTIFIER.\n"
            f"Identifier name: {nome}.\n"
            f"Identifier name repeated: {nome}.\n"
            f"Identifier type: {tipo}.\n"
            f"Identifier ID: {ident_id}.\n\n"
            f"This identifier refers exclusively to the statistical concept '{nome}'. "
            f"It is part of Brazilian socioeconomic and labor statistics produced by IBGE. "
            f"It must be used only for indicators directly related to '{nome}'. "
            f"It is NOT related to environmental, ecological, climate, or natural resource topics "
            f"unless such terms explicitly appear in the identifier name."
        )

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "tipo": tipo,
                    "id": ident_id,
                    "nome": nome,
                    "source": "ibge_identificador"
                }
            )
        )

    logger.info(f"{len(documents)} identificadores processados com sucesso")
    return documents


def add_documents_to_vector_store(
    documents: List[Document],
    batch_size: int = 20
) -> None:
    """
    Gera embeddings explicitamente e adiciona ao vector store
    com IDs técnicos únicos.
    """

    kb = get_ibge_knowledge_base()

    total_batches = (len(documents) + batch_size - 1) // batch_size
    logger.info(
        f"Iniciando adição de {len(documents)} documentos "
        f"em {total_batches} lotes"
    )

    success = 0
    errors = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = i // batch_size + 1

        texts = [doc.page_content for doc in batch]

        # 🔥 GERA EMBEDDINGS EXPLICITAMENTE
        embeddings = embedding_model.embed_documents(texts)

        ids = []
        for idx, doc in enumerate(batch):
            tipo = doc.metadata["tipo"]
            ident_id = doc.metadata["id"]
            unique_id = f"ibge_{tipo}_{ident_id}_{batch_num}_{idx}"
            ids.append(unique_id)

        try:
            kb.vector_store.add_embeddings(
                texts=texts,
                embeddings=embeddings,
                metadatas=[doc.metadata for doc in batch],
                ids=ids
            )

            success += len(batch)
            logger.info(
                f"✓ Lote {batch_num}/{total_batches} "
                f"({len(batch)} documentos)"
            )

        except Exception as e:
            errors += len(batch)
            logger.error(
                f"✗ Erro no lote {batch_num}/{total_batches}: {e}"
            )

    logger.info("=" * 60)
    logger.info("Resumo da carga:")
    logger.info(f"  Total:   {len(documents)}")
    logger.info(f"  Sucesso: {success}")
    logger.info(f"  Erros:   {errors}")
    logger.info("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("Carga de Identificadores IBGE (Multi-Tipo + Embeddings)")
    print("=" * 60 + "\n")

    csv_path = "src/memory/pgvector/identificadores.csv"

    try:
        if not os.path.exists(csv_path):
            print(f"❌ CSV não encontrado: {csv_path}")
            return

        kb = get_ibge_knowledge_base()
        if kb.check_collection_exists():
            resp = input(
                "\n⚠ A coleção já contém documentos. "
                "Deseja continuar? (s/n): "
            )
            if resp.lower() != "s":
                print("Operação cancelada.")
                return

        print(f"\n📊 1. Carregando identificadores: {csv_path}")
        documents = load_identificadores_csv(csv_path)

        if not documents:
            print("❌ Nenhum documento gerado.")
            return

        print(f"\n🚀 2. Gerando embeddings e inserindo no vector store...")
        add_documents_to_vector_store(documents)

        print("\n✅ Identificadores IBGE carregados com sucesso!")

    except Exception as e:
        logger.error("Erro inesperado", exc_info=True)
        print(f"\n❌ Erro: {e}")


if __name__ == "__main__":
    main()
