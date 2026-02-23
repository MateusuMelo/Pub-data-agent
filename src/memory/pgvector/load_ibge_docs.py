import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

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


def load_agregados_csv(csv_file_path: str) -> List[Document]:
    """
    Carrega agregados IBGE com conteúdo semântico OTIMIZADO.

    CSV esperado:
        pesquisa_id,pesquisa_nome,agregado_id,agregado_nome
        D5,Áreas Urbanizadas,8418,"Áreas urbanizadas, Loteamento vazio, Área total mapeada e Subcategorias"
        CL,Cadastro Central de Empresas,1685,"Unidades locais, empresas e outras organizações..."
    """

    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_file_path}")

    logger.info(f"📂 Carregando agregados IBGE de: {csv_file_path}")

    df = pd.read_csv(csv_file_path)

    # Log inicial do CSV
    logger.info(f"📊 CSV carregado: {len(df)} registros")

    # Contagem por pesquisa
    pesquisa_counts = df['pesquisa_nome'].value_counts()
    logger.info(f"📋 Pesquisas encontradas: {len(pesquisa_counts)}")
    for pesquisa, count in pesquisa_counts.head(10).items():
        logger.info(f"  - {pesquisa}: {count} agregados")

    # Validar colunas obrigatórias
    required_columns = {"pesquisa_id", "pesquisa_nome", "agregado_id", "agregado_nome"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV está faltando colunas obrigatórias: {missing}")

    documents: List[Document] = []

    for _, row in df.iterrows():
        pesquisa_id = str(row["pesquisa_id"]).strip()
        pesquisa_nome = str(row["pesquisa_nome"]).strip()
        agregado_id = str(row["agregado_id"]).strip()
        agregado_nome = str(row["agregado_nome"]).strip()

        # Pular linhas incompletas
        if not pesquisa_id or not pesquisa_nome or not agregado_id or not agregado_nome:
            logger.warning(f"⚠️  Linha ignorada - dados incompletos: {row.to_dict()}")
            continue

        # 🎯 CONTEÚDO SEMÂNTICO OTIMIZADO PARA BUSCA
        page_content = (
            # 1. NOME DO AGREGADO como foco principal
            f"AGREGADO IBGE: {agregado_nome}.\n"
            f"Nome completo do agregado: {agregado_nome}.\n"
            f"Conjunto de dados estatísticos: {agregado_nome}.\n"

            f"Este agregado pertence à pesquisa '{pesquisa_nome}' (código {pesquisa_id}) do IBGE.\n"

            f"Use o código {agregado_id} para acessar os dados agregados sobre {agregado_nome.lower()} "
            f"da pesquisa {pesquisa_id} ({pesquisa_nome}) do Instituto Brasileiro de Geografia e Estatística."
        )

        # Criar documento
        doc = Document(
            page_content=page_content,
            metadata={
                "pesquisa_id": pesquisa_id,
                "pesquisa_nome": pesquisa_nome,
                "id": agregado_id,
                "nome": agregado_nome,
                "source": "ibge_agregado",  # Alterado para diferenciar
                "document_type": "aggregate",
                "origin": "IBGE"
            }
        )

        documents.append(doc)

        # Log dos primeiros documentos
        if len(documents) <= 3:
            logger.debug(f"📄 Agregado criado: {pesquisa_id}:{agregado_id} - {agregado_nome[:30]}...")

    logger.info(f"✅ {len(documents)} documentos de agregados criados com sucesso")
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

            # Criar IDs únicos baseados no tipo de documento
            ids = []
            for idx, doc in enumerate(batch):
                source_type = doc.metadata.get("source", "unknown")

                if source_type == "ibge_identificador":
                    tipo = doc.metadata["tipo"]
                    ident_id = doc.metadata["id"]
                    unique_id = f"ibge_ident_{tipo}_{ident_id}_b{batch_num:03d}_{idx:03d}"
                elif source_type == "ibge_agregado":
                    agregado_id = doc.metadata["id"]
                    pesquisa_id = doc.metadata["pesquisa_id"]
                    unique_id = f"ibge_agreg_{pesquisa_id}_{agregado_id}_b{batch_num:03d}_{idx:03d}"
                else:
                    unique_id = f"ibge_unk_{batch_num:03d}_{idx:03d}"

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
            raise e
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

        # Distribuição por tipo de fonte
        cursor.execute("""
                       SELECT cmetadata ->>'source' as fonte, cmetadata ->>'document_type' as tipo_doc, COUNT (*) as quantidade
                       FROM langchain_pg_embedding
                       WHERE collection_id = (
                           SELECT uuid FROM langchain_pg_collection
                           WHERE name = %s
                           )
                       GROUP BY cmetadata->>'source', cmetadata->>'document_type'
                       ORDER BY quantidade DESC
                       """, (kb.vector_store.collection_name,))

        logger.info("  Distribuição por fonte e tipo:")
        resultados = cursor.fetchall()
        for fonte, tipo_doc, quantidade in resultados:
            logger.info(f"    - {fonte} ({tipo_doc}): {quantidade}")

        # Para identificadores: distribuição por tipo
        cursor.execute("""
                       SELECT cmetadata ->>'tipo' as tipo, COUNT (*) as quantidade
                       FROM langchain_pg_embedding
                       WHERE collection_id = (
                           SELECT uuid FROM langchain_pg_collection
                           WHERE name = %s
                           )
                         AND cmetadata->>'source' = 'ibge_identificador'
                       GROUP BY cmetadata->>'tipo'
                       ORDER BY quantidade DESC
                       """, (kb.vector_store.collection_name,))

        logger.info("  Identificadores por tipo:")
        tipos = cursor.fetchall()
        for tipo, quantidade in tipos:
            logger.info(f"    - {tipo}: {quantidade}")

        # Para agregados: distribuição por pesquisa
        cursor.execute("""
                       SELECT cmetadata ->>'pesquisa_nome' as pesquisa, COUNT (*) as quantidade
                       FROM langchain_pg_embedding
                       WHERE collection_id = (
                           SELECT uuid FROM langchain_pg_collection
                           WHERE name = %s
                           )
                         AND cmetadata->>'source' = 'ibge_agregado'
                       GROUP BY cmetadata->>'pesquisa_nome'
                       ORDER BY quantidade DESC
                           LIMIT 10
                       """, (kb.vector_store.collection_name,))

        logger.info("  Top 10 pesquisas com mais agregados:")
        pesquisas = cursor.fetchall()
        for pesquisa, quantidade in pesquisas:
            logger.info(f"    - {pesquisa}: {quantidade}")

        # Amostra de documentos
        cursor.execute("""
                       SELECT cmetadata ->>'source' as fonte, COALESCE (cmetadata->>'nome', cmetadata->>'agregado_nome') as nome, COALESCE (cmetadata->>'id', cmetadata->>'agregado_id') as codigo, LENGTH (document) as tamanho
                       FROM langchain_pg_embedding
                       WHERE collection_id = (
                           SELECT uuid FROM langchain_pg_collection
                           WHERE name = %s
                           )
                       ORDER BY RANDOM()
                           LIMIT 5
                       """, (kb.vector_store.collection_name,))

        logger.info("  Amostra aleatória de documentos:")
        amostras = cursor.fetchall()
        for fonte, nome, codigo, tamanho in amostras:
            logger.info(f"    - {fonte}: {codigo} - {nome} ({tamanho} chars)")

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
    print("📊 CARGA DE DADOS IBGE - VECTOR STORE")
    print("=" * 60 + "\n")

    # Caminhos dos CSVs
    identificadores_path = "data/identificadores_ibge.csv"
    agregados_path = "data/agregados_ibge.csv"

    # Caminhos alternativos para identificadores
    possible_ident_paths = [
        identificadores_path,
        "src/memory/pgvector/identificadores.csv",
        "data/identificadores.csv",
        "../data/identificadores_ibge.csv",
        str(Path(__file__).parent.parent.parent / "data" / "identificadores_ibge.csv")
    ]

    # Caminhos alternativos para agregados
    possible_agreg_paths = [
        agregados_path,
        "src/memory/pgvector/agregados_ibge.csv",
        "data/agregados.csv",
        "../data/agregados_ibge.csv",
        str(Path(__file__).parent.parent.parent / "data" / "agregados_ibge.csv")
    ]

    # Encontrar arquivos
    ident_found = None
    for path in possible_ident_paths:
        if os.path.exists(path):
            ident_found = path
            break

    agreg_found = None
    for path in possible_agreg_paths:
        if os.path.exists(path):
            agreg_found = path
            break

    if not ident_found and not agreg_found:
        logger.error("❌ Nenhum arquivo CSV encontrado nos caminhos:")
        logger.error("   Identificadores:")
        for path in possible_ident_paths:
            logger.error(f"     - {path}")
        logger.error("   Agregados:")
        for path in possible_agreg_paths:
            logger.error(f"     - {path}")
        return

    # Mostrar arquivos encontrados
    if ident_found:
        logger.info(f"📂 Arquivo de identificadores: {ident_found}")
    if agreg_found:
        logger.info(f"📂 Arquivo de agregados: {agreg_found}")

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

        all_documents = []

        # 2. Carregar identificadores (se arquivo existir)
        if ident_found:
            print(f"\n📥 1. Carregando identificadores do CSV...")
            ident_docs = load_identificadores_csv(ident_found)
            all_documents.extend(ident_docs)
            logger.info(f"✅ {len(ident_docs)} identificadores carregados")
        else:
            logger.warning("⚠️  Arquivo de identificadores não encontrado, pulando...")

        # 3. Carregar agregados (se arquivo existir)
        if agreg_found:
            print(f"\n📥 2. Carregando agregados do CSV...")
            agreg_docs = load_agregados_csv(agreg_found)
            all_documents.extend(agreg_docs)
            logger.info(f"✅ {len(agreg_docs)} agregados carregados")
        else:
            logger.warning("⚠️  Arquivo de agregados não encontrado, pulando...")

        if not all_documents:
            logger.error("❌ Nenhum documento foi gerado dos CSVs")
            return

        # 4. Adicionar ao vector store
        print(f"\n⚡ 3. Gerando embeddings e inserindo no vector store...")
        add_documents_to_vector_store(all_documents)

        # 5. Verificar carga
        print(f"\n🔍 4. Verificando carga...")
        verify_collection()

        print(f"\n✅ CARGA CONCLUÍDA COM SUCESSO!")
        print(f"   Total de documentos processados: {len(all_documents)}")

        # Resumo por tipo
        ident_count = sum(1 for doc in all_documents if doc.metadata.get("source") == "ibge_identificador")
        agreg_count = sum(1 for doc in all_documents if doc.metadata.get("source") == "ibge_agregado")

        if ident_count > 0:
            print(f"   - Identificadores: {ident_count}")
        if agreg_count > 0:
            print(f"   - Agregados: {agreg_count}")

    except Exception as e:
        logger.error("❌ ERRO NO PROCESSAMENTO", exc_info=True)
        print(f"\n💥 Erro fatal: {e}")


if __name__ == "__main__":
    main()