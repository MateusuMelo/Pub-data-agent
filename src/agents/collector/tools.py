from typing import Dict, Any, List, Union

import pandas as pd
import requests
from langchain.tools import tool

from src.memory.knowledge.vector_store import get_ibge_knowledge_base

# ======================================================
# Knowledge base (DOCUMENTATION ONLY)
# ======================================================

def _search_kb(
        query: str,
        *,
        tipo_filtro: str | None = None,  # Renomeei para ficar claro
        k: int = 10
):
    """
    Busca semântica otimizada para identificadores IBGE.
    """

    # 🔥 MELHORAR A QUERY ADICIONANDO CONTEXTO
    enhanced_query = (
        f"IBGE identificador {query} "
        f"estatísticas brasileiras {query}"
    )

    kb = get_ibge_knowledge_base()

    # Aplicar filtro se especificado
    if tipo_filtro:
        results = kb.vector_store.similarity_search_with_score(
            enhanced_query,
            k=k * 2,  # Busca mais para filtrar depois
            filter={"tipo": tipo_filtro}
        )
    else:
        results = kb.vector_store.similarity_search_with_score(
            enhanced_query,
            k=k
        )

    return results


@tool
def ibge_assunto_id_search(query: str) -> str:
    """
    Busca identificadores IBGE pelo NOME ou conceito relacionado.
    Retorna código e nome dos identificadores encontrados.

    Exemplo:
    - "PIB" → retorna identificadores relacionados a Produto Interno Bruto
    - "população" → retorna identificadores demográficos
    - "desemprego" → retorna identificadores de mercado de trabalho
    """

    # Buscar identificadores (focar em 'assunto' primeiro)
    results = _search_kb(
        query=query,
        tipo_filtro="assunto",  # Você pode ajustar ou remover o filtro
        k=15
    )

    if not results:
        # Tentar sem filtro se não encontrar com filtro
        results = _search_kb(
            query=query,
            tipo_filtro=None,
            k=10
        )

    if not results:
        return "Nenhum identificador IBGE encontrado para esta consulta."

    # Formatar resultados
    lines = []
    for doc, score in results:
        ident_id = doc.metadata.get("id", "N/A")
        nome = doc.metadata.get("nome", "N/A")
        tipo = doc.metadata.get("tipo", "N/A")

        lines.append(f"{tipo.upper()}: {ident_id} | {nome}")

    return "\n".join(lines)


@tool
def ibge_classificacao_id_search(query: str) -> str:
    """
    Busca identificadores IBGE pelo NOME ou conceito relacionado.
    Retorna código e nome dos identificadores encontrados.

    Exemplo:
    - "CNAE" → retorna classificações de atividades econômicas
    - "CID" → retorna classificações internacionais de doenças
    - "OCUPACAO" → retorna classificações ocupacionais
    """

    # Buscar identificadores com filtro 'classificacao'
    results = _search_kb(
        query=query,
        tipo_filtro="classificacao",  # Filtro específico para classificações
        k=15
    )

    if not results:
        # Tentar sem filtro se não encontrar com filtro específico
        results = _search_kb(
            query=query,
            tipo_filtro=None,
            k=10
        )

    if not results:
        return "Nenhum identificador IBGE de classificação encontrado para esta consulta."

    # Formatar resultados
    lines = []
    for doc, score in results:
        ident_id = doc.metadata.get("id", "N/A")
        nome = doc.metadata.get("nome", "N/A")
        tipo = doc.metadata.get("tipo", "N/A")

        lines.append(f"{tipo.upper()}: {ident_id} | {nome}")

    return "\n".join(lines)

@tool
def ibge_nivel_geografico_id_search(query: str) -> dict[str, Any] | str:
    """
    Busca ID geográfico no CSV. Query deve ser o ID.
    """
    try:
        df = pd.read_csv('data/identificadores.csv')
        query = query.strip()

        # Busca direta
        for _, row in df.iterrows():
            if str(row.get('id', '')) == query:
                nome = row.get('nome', 'N/A')
                tipo = row.get('tipo', 'N/A')
                return {
                    "id": row['id'],
                    "nivel_geografico": nome,
                }

        return f"ID {query} não encontrado."

    except:
        return "Erro ao acessar dados."

@tool
def ibge_periodicidade_id_search(query: str) -> str:
    """
    Discover IBGE periodicidade identifiers.
    """
    docs = _search_kb(query, source="periodicidade")
    return "\n".join(d.page_content for d in docs)


@tool
def ibge_periodo_id_search(query: str) -> str:
    """
    Discover IBGE period identifiers.
    """
    docs = _search_kb(query, source="periodo")
    return "\n".join(d.page_content for d in docs)


@tool
def ibge_variavel_id_search(query: str) -> str:
    """
    Discover IBGE variavel identifiers.
    """
    docs = _search_kb(query, source="variavel")
    return "\n".join(d.page_content for d in docs)


@tool
def ibge_documentation_search(query: str) -> str:
    """
    General IBGE documentation search.
    Used to understand concepts and API usage.
    """
    docs = _search_kb(query)
    return "\n\n".join(d.page_content for d in docs)


@tool
def ibge_documentation_search(query: str) -> str:
    """
    General IBGE documentation search.
    Used to understand concepts and API usage.
    """
    docs = _search_kb(query, source='IBGE_Swagger_API')
    return "\n\n".join(d.page_content for d in docs)


# ======================================================
# Secure execution tool (PARAMETERS ONLY)
# ======================================================

@tool
def ibge_agregados_request(
        assunto_id: int,
) -> Dict[str, Any]:
    """
    Discover IBGE aggregates related to a given ASSUNTO.

    This tool DOES NOT fetch time series data.
    It ONLY lists available aggregates linked to the assunto.

    Used internally:
    /v3/agregados?assunto={assunto_id}
    """

    base_url = "https://servicodados.ibge.gov.br/api"
    endpoint = f"/v3/agregados"
    params = {"assunto": assunto_id}

    url = f"{base_url}{endpoint}"

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    raw = response.json()

    temas: List[Dict[str, Any]] = []

    for tema in raw:
        temas.append({
            "tema_id": tema.get("id"),
            "tema_nome": tema.get("nome"),
            "agregados": [
                {
                    "agregado_id": agg.get("id"),
                    "agregado_nome": agg.get("nome"),
                }
                for agg in tema.get("agregados", [])
            ]
        })

    return {
        "success": True,
        "source": "IBGE / SIDRA",
        "tipo": "descoberta_de_agregados",
        "assunto_id": assunto_id,
        "endpoint": endpoint,
        "params": params,
        "temas_encontrados": temas,
        "metadata": {
            "total_temas": len(temas),
            "total_agregados": sum(len(t["agregados"]) for t in temas),
        }
    }


@tool
def ibge_agregado_metadados_request(
        agregado_id: int,
) -> Dict[str, Any]:
    """
    Obtém metadados COMPLETOS de um agregado IBGE, incluindo períodos disponíveis.

    Combina duas requisições em uma:
    1. Metadados do agregado
    2. Períodos disponíveis

    Args:
        agregado_id: ID numérico do agregado IBGE (obrigatório)
    """

    base_url = "https://servicodados.ibge.gov.br/api"

    # 1. Buscar metadados do agregado
    url_metadados = f"{base_url}/v3/agregados/{agregado_id}/metadados"

    try:
        response_metadados = requests.get(url_metadados, timeout=30)
        response_metadados.raise_for_status()
    except Exception as e:
        return {
            "success": False,
            "error": f"Erro ao buscar metadados: {str(e)}",
            "agregado_id": agregado_id
        }

    raw_metadados = response_metadados.json()

    # A API retorna uma lista com um item
    if isinstance(raw_metadados, list) and len(raw_metadados) > 0:
        agregado_data = raw_metadados[0]
    else:
        agregado_data = raw_metadados

    # 2. Buscar períodos disponíveis
    url_periodos = f"{base_url}/v3/agregados/{agregado_id}/periodos"

    try:
        response_periodos = requests.get(url_periodos, timeout=30)
        response_periodos.raise_for_status()
        periodos_data = response_periodos.json()
    except Exception as e:
        # Se der erro nos períodos, ainda retorna os metadados
        periodos_data = []

    # Extrair informações básicas dos metadados
    metadados = {
        "agregado_id": agregado_data.get("id"),
        "agregado_nome": agregado_data.get("nome"),
        "url_sidra": agregado_data.get("URL"),
        "pesquisa": agregado_data.get("pesquisa"),
        "assunto": agregado_data.get("assunto"),
    }

    # Periodicidade
    periodicidade = agregado_data.get("periodicidade", {})
    if periodicidade:
        metadados["periodicidade"] = {
            "frequencia": periodicidade.get("frequencia"),
            "inicio": periodicidade.get("inicio"),
            "fim": periodicidade.get("fim"),
        }

    # Níveis territoriais
    niveis_territoriais = agregado_data.get("nivelTerritorial", {})
    if niveis_territoriais:
        metadados["nivelTerritorial"] = {
            "Administrativo": niveis_territoriais.get("Administrativo", []),
            "Especial": niveis_territoriais.get("Especial", []),
            "IBGE": niveis_territoriais.get("IBGE", []),
        }

    # Variáveis
    variaveis = agregado_data.get("variaveis", [])
    if variaveis:
        metadados["variaveis"] = variaveis

    # Classificações
    classificacoes = agregado_data.get("classificacoes", [])
    if classificacoes:
        metadados["classificacoes"] = classificacoes

    # Processar períodos
    periodos_formatados = []
    if periodos_data and isinstance(periodos_data, list):
        for periodo in periodos_data:
            periodos_formatados.append({
                "periodo_id": periodo.get("id"),
                "representacoes_textuais": periodo.get("literals", []),
                "ultima_modificacao": periodo.get("modificacao"),
            })

    return {
        "success": True,
        "source": "IBGE / SIDRA",
        "tipo": "metadados_completos_agregado",
        "agregado_id": agregado_id,
        "endpoints_utilizados": [
            f"/v3/agregados/{agregado_id}/metadados",
            f"/v3/agregados/{agregado_id}/periodos"
        ],
        "metadados": metadados,
        "periodos_disponiveis": {
            "total_periodos": len(periodos_formatados),
            "periodos": periodos_formatados,
            "amostra_ids": [p["periodo_id"] for p in periodos_formatados[:5]] if periodos_formatados else []
        },
        "metadata": {
            "total_variaveis": len(variaveis),
            "total_classificacoes": len(classificacoes),
            "total_periodos": len(periodos_formatados),
        }
    }


@tool
def ibge_agregado_dados_request(
        agregado: int,
        periodo: Union[int, List[int]],
        variavel: Union[int, List[int]],
        territorio: Union[str, List[str]] = "N1[all]",
        classificacao: Union[int, Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Obtém dados de um agregado IBGE.

    Args:
        agregado: ID do agregado (int)
        periodo: Período como inteiro ou lista de inteiros
        variavel: ID da variável como inteiro ou lista de inteiros
        territorio: Localidade(s) como string
        classificacao: Classificação como int ou dicionário
    """

    base_url = "https://servicodados.ibge.gov.br/api"

    # Formatar período
    if isinstance(periodo, list):
        periodo_str = ",".join(str(p) for p in periodo)
    else:
        periodo_str = str(periodo)

    # Formatar variável
    if isinstance(variavel, list):
        variavel_str = ",".join(str(v) for v in variavel)
    else:
        variavel_str = str(variavel)

    endpoint = f"/v3/agregados/{agregado}/periodos/{periodo_str}/variaveis/{variavel_str}"
    url = f"{base_url}{endpoint}"

    # Preparar parâmetros
    params = {}

    if territorio:
        if isinstance(territorio, list):
            params["localidades"] = ",".join(territorio)
        else:
            params["localidades"] = territorio

    if classificacao:
        if isinstance(classificacao, dict):
            # Converter dicionário para formato IBGE
            classificacao_strs = []
            for classif_id, categorias in classificacao.items():
                if categorias is None or categorias == "all":
                    # Se for None ou string "all", usar [all]
                    classificacao_strs.append(f"{classif_id}[all]")
                elif isinstance(categorias, list):
                    if not categorias:  # Lista vazia
                        classificacao_strs.append(f"{classif_id}[all]")
                    else:
                        cats = ",".join(str(c) for c in categorias)
                        classificacao_strs.append(f"{classif_id}[{cats}]")
                elif isinstance(categorias, str):
                    if categorias.lower() == "all":
                        classificacao_strs.append(f"{classif_id}[all]")
                    else:
                        classificacao_strs.append(f"{classif_id}[{categorias}]")
                else:
                    # Para int, float, etc.
                    classificacao_strs.append(f"{classif_id}[{categorias}]")
            params["classificacao"] = "|".join(classificacao_strs)
        else:
            # Se for string, int, etc.
            params["classificacao"] = str(classificacao)
    # Fazer requisição
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()
