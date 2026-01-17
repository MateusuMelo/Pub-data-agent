import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
import psycopg2

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.memory.knowledge.vector_store import get_ibge_knowledge_base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 🔥 EMBEDDING MODEL (OLLAMA)
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)


def reset_collection() -> None:
    """
    Limpa completamente a coleção existente no PostgreSQL.
    Útil para recomeçar do zero.
    """
    logger.warning("⚠️  LIMPANDO COLEÇÃO EXISTENTE...")

    try:
        kb = get_ibge_knowledge_base()

        # Usar o método get_connection_string
        conn = psycopg2.connect(kb.vector_store.get_connection_string())
        cursor = conn.cursor()

        # 1. Deletar embeddings da coleção
        cursor.execute("""
                       DELETE
                       FROM langchain_pg_embedding
                       WHERE collection_id = (SELECT uuid
                                              FROM langchain_pg_collection
                                              WHERE name = %s)
                       """, (kb.vector_store.collection_name,))

        # 2. Deletar a coleção
        cursor.execute("""
                       DELETE
                       FROM langchain_pg_collection
                       WHERE name = %s
                       """, (kb.vector_store.collection_name,))

        conn.commit()
        logger.info(f"✅ Coleção '{kb.vector_store.collection_name}' completamente limpa.")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"❌ Erro ao limpar coleção: {e}")
        raise


def load_identificadores_csv(csv_file_path: str) -> List[Document]:
    """
    Carrega identificadores IBGE com conteúdo semântico OTIMIZADO.
    Foco em buscas pelo NOME dos identificadores.

    CSV esperado:
        tipo,id,nome
        assunto,1,Nascidos vivos
        assunto,100,Balneabilidade
        ...
    """

    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_file_path}")

    logger.info(f"📂 Carregando identificadores IBGE de: {csv_file_path}")

    df = pd.read_csv(csv_file_path)

    # Log inicial do CSV
    logger.info(f"📊 CSV carregado: {len(df)} registros")
    logger.info(f"📋 Tipos únicos encontrados: {df['tipo'].unique().tolist()}")

    # Contagem por tipo
    tipo_counts = df['tipo'].value_counts()
    for tipo, count in tipo_counts.items():
        logger.info(f"  - {tipo}: {count} registros")

    # Validar colunas obrigatórias
    required_columns = {"tipo", "id", "nome"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV está faltando colunas obrigatórias: {missing}")

    documents: List[Document] = []

    for _, row in df.iterrows():
        tipo = str(row["tipo"]).strip()
        ident_id = str(row["id"]).strip()
        nome = str(row["nome"]).strip()

        # Pular linhas incompletas
        if not tipo or not ident_id or not nome:
            logger.warning(f"⚠️  Linha ignorada - dados incompletos: {row.to_dict()}")
            continue

        # 🎯 CONTEÚDO SEMÂNTICO OTIMIZADO PARA BUSCA
        page_content = (
            # 1. NOME como foco principal (várias formas)
            f"IDENTIFICADOR IBGE: {nome}.\n"
            f"Nome completo: {nome}.\n"
            f"Conceito estatístico: {nome}.\n"

            # 2. Descrição contextual
            f"Este é um {tipo} das estatísticas oficiais do Brasil produzidas pelo IBGE.\n"

            # 3. Sinônimos e termos relacionados
            f"Termos relacionados: {nome.lower()}, dados de {nome.lower()}, "
            f"estatísticas de {nome.lower()}, indicadores de {nome.lower()}.\n"

            # 4. Metadados explícitos
            f"Código único: {ident_id}.\n"
            f"Categoria: {tipo}.\n"

            # 5. Instrução de uso
            f"Use este código {ident_id} para consultar dados sobre {nome.lower()} "
            f"nas bases estatísticas do Instituto Brasileiro de Geografia e Estatística."
        )

        # Criar documento
        doc = Document(
            page_content=page_content,
            metadata={
                "tipo": tipo,
                "id": ident_id,
                "nome": nome,
                "source": "ibge_identificador",
                "document_type": "identifier",
                "origin": "IBGE"
            }
        )

        documents.append(doc)

        # Log dos primeiros documentos
        if len(documents) <= 3:
            logger.debug(f"📄 Documento criado: {tipo}:{ident_id} - {nome[:30]}...")

    logger.info(f"✅ {len(documents)} documentos criados com sucesso")
    return documents


def add_documents_to_vector_store(
        documents: List[Document],
        batch_size: int = 50
) -> None:
    """
    Adiciona documentos ao vector store com embeddings pré-gerados.
    """

    kb = get_ibge_knowledge_base()

    total_batches = (len(documents) + batch_size - 1) // batch_size
    logger.info(f"🚀 Inserindo {len(documents)} documentos em {total_batches} lotes")

    success_count = 0
    error_count = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = i // batch_size + 1

        logger.info(f"📦 Processando lote {batch_num}/{total_batches} ({len(batch)} documentos)")

        try:
            # Extrair textos
            texts = [doc.page_content for doc in batch]

            # Gerar embeddings explicitamente
            logger.debug(f"  Gerando embeddings para lote {batch_num}...")
            embeddings = embedding_model.embed_documents(texts)

            # Criar IDs únicos
            ids = []
            for idx, doc in enumerate(batch):
                tipo = doc.metadata["tipo"]
                ident_id = doc.metadata["id"]
                unique_id = f"ibge_{tipo}_{ident_id}_b{batch_num:03d}_{idx:03d}"
                ids.append(unique_id)

            # Adicionar ao vector store
            kb.vector_store.add_embeddings(
                texts=texts,
                embeddings=embeddings,
                metadatas=[doc.metadata for doc in batch],
                ids=ids
            )

            success_count += len(batch)
            logger.info(f"  ✅ Lote {batch_num} inserido com sucesso")

        except Exception as e:
            error_count += len(batch)
            logger.error(f"  ❌ Erro no lote {batch_num}: {str(e)}")
            # Continuar com os próximos lotes

    # Resumo final
    logger.info("=" * 60)
    logger.info("📊 RESUMO DA CARGA:")
    logger.info(f"  Total de documentos: {len(documents)}")
    logger.info(f"  Inseridos com sucesso: {success_count}")
    logger.info(f"  Com erro: {error_count}")

    if error_count == 0:
        logger.info("🎉 TODOS os documentos foram carregados com sucesso!")
    else:
        logger.warning(f"⚠️  {error_count} documentos não foram carregados")
    logger.info("=" * 60)


def verify_collection() -> None:
    """
    Verifica se os documentos foram carregados corretamente.
    """
    logger.info("🔍 VERIFICANDO COLEÇÃO...")

    try:
        kb = get_ibge_knowledge_base()

        # Usar o método get_connection_string
        conn = psycopg2.connect(kb.vector_store.get_connection_string())
        cursor = conn.cursor()

        # Verificar se a coleção existe
        cursor.execute("""
                       SELECT EXISTS (SELECT 1
                                      FROM langchain_pg_collection
                                      WHERE name = %s)
                       """, (kb.vector_store.collection_name,))

        collection_exists = cursor.fetchone()[0]

        if not collection_exists:
            logger.warning("⚠️  Coleção não encontrada no banco de dados")
            return

        # Contar documentos
        cursor.execute("""
                       SELECT COUNT(*) as total
                       FROM langchain_pg_embedding
                       WHERE collection_id = (SELECT uuid
                                              FROM langchain_pg_collection
                                              WHERE name = %s)
                       """, (kb.vector_store.collection_name,))

        total_docs = cursor.fetchone()[0]
        logger.info(f"  Total de documentos na coleção: {total_docs}")

        # Distribuição por tipo
        cursor.execute("""
                       SELECT cmetadata ->>'tipo' as tipo, COUNT (*) as quantidade
                       FROM langchain_pg_embedding
                       WHERE collection_id = (
                           SELECT uuid FROM langchain_pg_collection
                           WHERE name = %s
                           )
                       GROUP BY cmetadata->>'tipo'
                       ORDER BY quantidade DESC
                       """, (kb.vector_store.collection_name,))

        logger.info("  Distribuição por tipo:")
        tipos = cursor.fetchall()
        for tipo, quantidade in tipos:
            logger.info(f"    - {tipo}: {quantidade}")

        # Amostra de documentos
        cursor.execute("""
                       SELECT cmetadata ->>'nome' as nome, cmetadata->>'tipo' as tipo, cmetadata->>'id' as id, LENGTH (document) as tamanho
                       FROM langchain_pg_embedding
                       WHERE collection_id = (
                           SELECT uuid FROM langchain_pg_collection
                           WHERE name = %s
                           )
                           LIMIT 5
                       """, (kb.vector_store.collection_name,))

        logger.info("  Amostra de documentos:")
        amostras = cursor.fetchall()
        for nome, tipo, ident_id, tamanho in amostras:
            logger.info(f"    - {tipo}:{ident_id} - {nome} ({tamanho} chars)")

        cursor.close()
        conn.close()

        logger.info("✅ Verificação concluída")

    except Exception as e:
        logger.error(f"❌ Erro na verificação: {e}")


def check_collection_exists() -> bool:
    """
    Verifica se a coleção já existe no banco.
    """
    try:
        kb = get_ibge_knowledge_base()
        conn = psycopg2.connect(kb.vector_store.get_connection_string())
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT EXISTS (SELECT 1
                                      FROM langchain_pg_collection
                                      WHERE name = %s)
                       """, (kb.vector_store.collection_name,))

        exists = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return exists

    except Exception as e:
        logger.error(f"Erro ao verificar coleção: {e}")
        return False


def main():
    """
    Função principal para executar o carregamento completo.
    """
    print("\n" + "=" * 60)
    print("📊 CARGA DE IDENTIFICADORES IBGE - VECTOR STORE")
    print("=" * 60 + "\n")

    # Caminho do CSV
    csv_path = "data/identificadores_ibge.csv"

    # Caminhos alternativos
    possible_paths = [
        csv_path,
        "src/memory/pgvector/identificadores.csv",
        "data/identificadores.csv",
        "../data/identificadores_ibge.csv",
        str(Path(__file__).parent.parent.parent / "data" / "identificadores_ibge.csv")
    ]

    csv_found = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_found = path
            break

    if not csv_found:
        logger.error("❌ Nenhum arquivo CSV encontrado nos caminhos:")
        for path in possible_paths:
            logger.error(f"   - {path}")
        return

    logger.info(f"📂 Usando arquivo: {csv_found}")

    try:
        # 1. Verificar se coleção existe
        collection_exists = check_collection_exists()

        if collection_exists:
            print(f"\n⚠️  A coleção '{get_ibge_knowledge_base().vector_store.collection_name}' já existe.")
            print("   Você pode:")
            print("   1. Adicionar novos documentos à coleção existente")
            print("   2. Limpar e recriar a coleção (perde dados existentes)")
            print("   3. Sair")

            choice = input("\nEscolha (1/2/3): ").strip()

            if choice == "2":
                reset = input("⚠️  TEM CERTEZA? Isso apagará TODOS os dados. (s/n): ")
                if reset.lower() == 's':
                    reset_collection()
                    collection_exists = False  # Agora não existe mais
                else:
                    print("Operação cancelada.")
                    return
            elif choice == "3":
                print("Operação cancelada.")
                return
            # Se escolher 1, continua normalmente

        # 2. Carregar documentos do CSV
        print(f"\n📥 1. Carregando documentos do CSV...")
        documents = load_identificadores_csv(csv_found)

        if not documents:
            logger.error("❌ Nenhum documento foi gerado do CSV")
            return

        # 3. Adicionar ao vector store
        print(f"\n⚡ 2. Gerando embeddings e inserindo no vector store...")
        add_documents_to_vector_store(documents)

        # 4. Verificar carga
        print(f"\n🔍 3. Verificando carga...")
        verify_collection()

        print(f"\n✅ CARGA CONCLUÍDA COM SUCESSO!")
        print(f"   Total de documentos processados: {len(documents)}")

    except Exception as e:
        logger.error("❌ ERRO NO PROCESSAMENTO", exc_info=True)
        print(f"\n💥 Erro fatal: {e}")


if __name__ == "__main__":
    main()