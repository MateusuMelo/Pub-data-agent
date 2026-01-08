from typing import Optional, Dict, Any, List

import requests
from langchain.tools import tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

from src.memory.knowledge.vector_store import get_ibge_knowledge_base
# ======================================================
# Knowledge base (DOCUMENTATION ONLY)
# ======================================================

kb = get_ibge_knowledge_base()

def _search_kb(
    query: str,
    *,
    source: str | None = None,
    k: int = 10,
    score_threshold: float = 0.75
):
    """
    Semantic search with score filtering to avoid unrelated results.
    """

    if source:
        results = kb.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter={"tipo": source}
        )
    else:
        results = kb.vector_store.similarity_search_with_score(query, k=k)

    # 🔥 FILTRO POR RELEVÂNCIA REAL
    filtered = [
        (doc, score)
        for doc, score in results
        if score >= score_threshold
    ]

    return filtered


from langchain.tools import tool


@tool
def ibge_assunto_id_search(query: str) -> str:
    """
    Discover IBGE assunto identifiers.
    Returns ONLY individual IDs and names.
    No interpretation. No explanation.
    """

    results = _search_kb(
        query,
        source="assunto",
        k=500,
        score_threshold=0.78
    )

    if not results:
        return ""

    lines = []
    for doc, score in results:
        ident_id = doc.metadata.get("id")
        nome = doc.metadata.get("nome")

        if ident_id and nome:
            lines.append(f"{ident_id} | {nome}")

    return "\n".join(lines)


@tool
def ibge_classificacao_id_search(query: str) -> str:
    """
    Discover IBGE classificacao identifiers.
    """
    docs = _search_kb(query, source="classificacao")
    return "\n".join(d.page_content for d in docs)


@tool
def ibge_nivel_geografico_id_search(query: str) -> str:
    """
    Discover IBGE geographic level identifiers.
    """
    docs = _search_kb(query, source="nivel_geografico")
    return "\n".join(d.page_content for d in docs)

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
