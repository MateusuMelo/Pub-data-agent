import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.agents.state import AgentState
from src.agents.communicator.agent import communicator_agent

logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import re


def communicator_node(state: AgentState) -> AgentState:
    """
    Communicator node - transforma dados da API IBGE/SIDRA em DataFrame pandas.
    """
    logger.info("📊 Communicator node started - Transformando dados IBGE em pandas")

    # Obter estado atual
    execution_plan = state.get("execution_plan", [])
    current_step = state.get("current_step", 0)
    collector_data = state.get("data", {})
    analysis_data = state.get("analysis", {})
    messages = state.get("messages", [])

    # Extrair última pergunta do usuário
    user_question = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            user_question = msg.content.strip()
            break

    try:
        # Transformar dados IBGE em DataFrame pandas
        df, metadata = transform_ibge_to_dataframe(collector_data)

        # Gerar resposta estruturada
        response_data = {
            "dataframe_info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            },
            "preview": df.head(20).to_dict(orient='records'),
            "metadata": metadata,
            "summary": generate_dataframe_summary(df),
            "user_question": user_question,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Converter para JSON amigável
        final_response = json.dumps(response_data, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✅ Communicator node concluído. DataFrame shape: {df.shape}")

        # Preparar resultado estruturado
        communication_result = {
            "success": True,
            "response": final_response,
            "response_type": "dataframe_transformation",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": {
                "original_question": user_question,
                "dataframe_rows": df.shape[0],
                "dataframe_columns": df.shape[1],
                "variables_count": metadata.get("total_variaveis", 0),
                "localidades_count": metadata.get("total_localidades", 0),
                "periodos_count": metadata.get("total_periodos", 0)
            }
        }

        # Retornar estado atualizado
        return {
            "messages": messages + [AIMessage(content=final_response)],
            "execution_plan": execution_plan,
            "current_step": current_step + 1,
            "data": collector_data,
            "analysis": analysis_data,
            "dataframe": df.to_dict(orient='records'),  # DataFrame serializado
            "dataframe_metadata": {
                "shape": df.shape,
                "columns": list(df.columns),
                "index": list(df.index) if not df.index.empty else []
            },
            "answer": final_response,
            "communication_result": communication_result
        }

    except Exception as e:
        logger.exception("❌ Erro no communicator node")

        # Gerar resposta de erro
        error_response = {
            "erro": True,
            "mensagem": f"Erro ao transformar dados em DataFrame: {str(e)}",
            "tipo_erro": type(e).__name__,
            "sugestoes": [
                "Verifique se os dados seguem o formato IBGE/SIDRA",
                "Confirme se há dados disponíveis para análise",
                "Tente novamente com uma pergunta mais específica"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

        return {
            "messages": messages + [AIMessage(content=json.dumps(error_response, indent=2))],
            "execution_plan": execution_plan,
            "current_step": current_step + 1,
            "data": collector_data,
            "analysis": analysis_data,
            "answer": json.dumps(error_response, indent=2),
            "communication_result": {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }


def transform_ibge_to_dataframe(data: Union[Dict, List]) -> tuple[pd.DataFrame, Dict]:
    """
    Transforma dados da API IBGE/SIDRA em DataFrame pandas.

    Args:
        data: Dados no formato IBGE/SIDRA

    Returns:
        tuple: (DataFrame pandas, metadados)
    """
    # Se for uma lista, processar cada item
    if isinstance(data, list):
        all_rows = []
        for item in data:
            rows = flatten_ibge_structure(item)
            all_rows.extend(rows)
    elif isinstance(data, dict):
        all_rows = flatten_ibge_structure(data)
    else:
        raise ValueError(f"Formato de dados não suportado: {type(data)}")

    # Criar DataFrame
    if all_rows:
        df = pd.DataFrame(all_rows)
    else:
        # DataFrame vazio com colunas padrão
        df = pd.DataFrame(columns=[
            'variavel_id', 'variavel', 'unidade',
            'classificacao_id', 'classificacao', 'categoria_id', 'categoria',
            'localidade_id', 'localidade', 'nivel_geografico_id', 'nivel_geografico',
            'periodo', 'valor', 'valor_numerico'
        ])

    # Calcular metadados
    metadata = calculate_dataframe_metadata(df, all_rows)

    return df, metadata


def flatten_ibge_structure(data_item: Dict) -> List[Dict]:
    """
    Transforma estrutura hierárquica IBGE em linhas tabulares.

    Args:
        data_item: Item de dados no formato IBGE

    Returns:
        List: Lista de dicionários representando linhas
    """
    rows = []

    # Extrair informações básicas
    variable_id = data_item.get("id", "")
    variable_name = data_item.get("variavel", "")
    unit = data_item.get("unidade", "")
    resultados = data_item.get("resultados", [])

    # Se não houver resultados, criar linha básica
    if not resultados:
        rows.append({
            "variavel_id": variable_id,
            "variavel": variable_name,
            "unidade": unit,
            "classificacao_id": "",
            "classificacao": "",
            "categoria_id": "",
            "categoria": "",
            "localidade_id": "",
            "localidade": "",
            "nivel_geografico_id": "",
            "nivel_geografico": "",
            "periodo": "",
            "valor": "",
            "valor_numerico": None
        })
        return rows

    # Processar cada resultado
    for resultado in resultados:
        classificacoes = resultado.get("classificacoes", [])
        series = resultado.get("series", [])

        # Processar classificações
        classification_rows = []

        if classificacoes:
            for classificacao in classificacoes:
                class_id = classificacao.get("id", "")
                class_name = classificacao.get("nome", "")
                categoria = classificacao.get("categoria", {})

                # Extrair categoria (primeira chave-valor)
                cat_id, cat_name = "", ""
                if categoria and isinstance(categoria, dict):
                    items = list(categoria.items())
                    if items:
                        cat_id, cat_name = items[0]

                classification_rows.append({
                    "classificacao_id": class_id,
                    "classificacao": class_name,
                    "categoria_id": cat_id,
                    "categoria": cat_name
                })
        else:
            # Sem classificação
            classification_rows.append({
                "classificacao_id": "",
                "classificacao": "",
                "categoria_id": "",
                "categoria": ""
            })

        # Processar séries
        if series:
            for serie in series:
                localidade = serie.get("localidade", {})
                localidade_id = localidade.get("id", "")
                localidade_nome = localidade.get("nome", "")

                nivel = localidade.get("nivel", {})
                nivel_id = nivel.get("id", "") if nivel else ""
                nivel_nome = nivel.get("nome", "") if nivel else ""

                serie_data = serie.get("serie", {})

                # Se houver dados na série
                if serie_data:
                    for periodo, valor in serie_data.items():
                        # Tentar converter para numérico
                        valor_numerico = None
                        valor_str = str(valor)

                        try:
                            # Remover caracteres não numéricos exceto ponto e vírgula
                            clean_val = str(valor).replace(',', '.')
                            # Remover caracteres não numéricos exceto ponto, sinal negativo
                            clean_val = re.sub(r'[^\d\.\-]', '', clean_val)
                            if clean_val and clean_val != '.' and clean_val != '-':
                                valor_numerico = float(clean_val)
                        except (ValueError, TypeError):
                            valor_numerico = None

                        # Para cada combinação de classificação
                        for class_row in classification_rows:
                            rows.append({
                                "variavel_id": variable_id,
                                "variavel": variable_name,
                                "unidade": unit,
                                "classificacao_id": class_row["classificacao_id"],
                                "classificacao": class_row["classificacao"],
                                "categoria_id": class_row["categoria_id"],
                                "categoria": class_row["categoria"],
                                "localidade_id": localidade_id,
                                "localidade": localidade_nome,
                                "nivel_geografico_id": nivel_id,
                                "nivel_geografico": nivel_nome,
                                "periodo": str(periodo),
                                "valor": valor_str,
                                "valor_numerico": valor_numerico
                            })
                else:
                    # Série vazia
                    for class_row in classification_rows:
                        rows.append({
                            "variavel_id": variable_id,
                            "variavel": variable_name,
                            "unidade": unit,
                            "classificacao_id": class_row["classificacao_id"],
                            "classificacao": class_row["classificacao"],
                            "categoria_id": class_row["categoria_id"],
                            "categoria": class_row["categoria"],
                            "localidade_id": localidade_id,
                            "localidade": localidade_nome,
                            "nivel_geografico_id": nivel_id,
                            "nivel_geografico": nivel_nome,
                            "periodo": "",
                            "valor": "",
                            "valor_numerico": None
                        })
        else:
            # Sem séries
            for class_row in classification_rows:
                rows.append({
                    "variavel_id": variable_id,
                    "variavel": variable_name,
                    "unidade": unit,
                    "classificacao_id": class_row["classificacao_id"],
                    "classificacao": class_row["classificacao"],
                    "categoria_id": class_row["categoria_id"],
                    "categoria": class_row["categoria"],
                    "localidade_id": "",
                    "localidade": "",
                    "nivel_geografico_id": "",
                    "nivel_geografico": "",
                    "periodo": "",
                    "valor": "",
                    "valor_numerico": None
                })

    return rows


def calculate_dataframe_metadata(df: pd.DataFrame, rows: List[Dict]) -> Dict:
    """
    Calcula metadados do DataFrame.

    Args:
        df: DataFrame pandas
        rows: Lista de linhas originais

    Returns:
        Dict: Metadados
    """
    metadata = {
        "total_linhas": len(df),
        "total_colunas": len(df.columns) if not df.empty else 0,
        "total_variaveis": 0,
        "total_localidades": 0,
        "total_periodos": 0,
        "variaveis": [],
        "localidades": [],
        "periodos": [],
        "niveis_geograficos": [],
        "classificacoes": []
    }

    if df.empty:
        return metadata

    # Estatísticas básicas
    metadata.update({
        "total_variaveis": df['variavel'].nunique() if 'variavel' in df.columns else 0,
        "total_localidades": df['localidade'].nunique() if 'localidade' in df.columns else 0,
        "total_periodos": df['periodo'].nunique() if 'periodo' in df.columns else 0,
        "variaveis": df['variavel'].unique().tolist() if 'variavel' in df.columns else [],
        "localidades": df['localidade'].unique().tolist() if 'localidade' in df.columns else [],
        "periodos": sorted(df['periodo'].unique().tolist()) if 'periodo' in df.columns else [],
        "niveis_geograficos": df['nivel_geografico'].unique().tolist() if 'nivel_geografico' in df.columns else [],
        "classificacoes": df['classificacao'].unique().tolist() if 'classificacao' in df.columns else []
    })

    # Estatísticas de valores numéricos
    if 'valor_numerico' in df.columns:
        numeric_values = df['valor_numerico'].dropna()
        if not numeric_values.empty:
            metadata["estatisticas_numericas"] = {
                "contagem": len(numeric_values),
                "media": float(numeric_values.mean()),
                "mediana": float(numeric_values.median()),
                "minimo": float(numeric_values.min()),
                "maximo": float(numeric_values.max()),
                "desvio_padrao": float(numeric_values.std()),
                "soma": float(numeric_values.sum())
            }

    return metadata


def generate_dataframe_summary(df: pd.DataFrame) -> Dict:
    """
    Gera um resumo estatístico do DataFrame.

    Args:
        df: DataFrame pandas

    Returns:
        Dict: Resumo estatístico
    """
    summary = {
        "informacoes_gerais": {},
        "estatisticas_descritivas": {},
        "valores_unicos": {},
        "valores_faltantes": {}
    }

    if df.empty:
        summary["informacoes_gerais"] = {
            "mensagem": "DataFrame vazio",
            "shape": df.shape
        }
        return summary

    # Informações gerais
    summary["informacoes_gerais"] = {
        "shape": df.shape,
        "colunas": list(df.columns),
        "tipos_dados": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memoria_uso": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
    }

    # Valores únicos por coluna
    for col in df.columns:
        unique_count = df[col].nunique()
        summary["valores_unicos"][col] = {
            "quantidade": unique_count,
            "exemplos": df[col].dropna().unique().tolist()[:5] if unique_count > 0 else []
        }

    # Valores faltantes
    missing = df.isnull().sum()
    summary["valores_faltantes"] = {
        col: {
            "quantidade": int(missing[col]),
            "percentual": f"{(missing[col] / len(df)) * 100:.2f}%"
        }
        for col in df.columns
    }

    # Estatísticas descritivas para colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        desc_stats = df[numeric_cols].describe()
        summary["estatisticas_descritivas"] = {
            col: {
                "contagem": int(desc_stats[col]['count']),
                "media": float(desc_stats[col]['mean']),
                "desvio_padrao": float(desc_stats[col]['std']),
                "minimo": float(desc_stats[col]['min']),
                "25%": float(desc_stats[col]['25%']),
                "mediana": float(desc_stats[col]['50%']),
                "75%": float(desc_stats[col]['75%']),
                "maximo": float(desc_stats[col]['max'])
            }
            for col in numeric_cols
        }

    return summary


# Função auxiliar para lidar com diferentes formatos de dados
def normalize_ibge_data(data: Any) -> List[Dict]:
    """
    Normaliza dados IBGE para formato consistente.

    Args:
        data: Dados brutos da API

    Returns:
        List: Lista de itens no formato IBGE
    """
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Verificar se é um único item ou resposta da API
        if "variavel" in data or "resultados" in data:
            return [data]
        elif "data" in data:
            return data["data"]
        else:
            # Tentar encontrar dados em qualquer chave
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    if "variavel" in value[0] or "resultados" in value[0]:
                        return value
            return [data]
    else:
        return []


# Versão alternativa se quiser manter também a resposta em texto
def generate_human_readable_summary(df: pd.DataFrame, metadata: Dict) -> str:
    """
    Gera um resumo em texto legível para humanos.

    Args:
        df: DataFrame pandas
        metadata: Metadados dos dados

    Returns:
        str: Resumo em texto
    """
    if df.empty:
        return "Não há dados disponíveis para análise."

    summary_parts = []

    # Informações básicas
    summary_parts.append(f"## 📊 Resumo dos Dados")
    summary_parts.append(f"")
    summary_parts.append(f"**Dimensões:** {df.shape[0]} linhas × {df.shape[1]} colunas")
    summary_parts.append(f"**Variáveis analisadas:** {metadata.get('total_variaveis', 0)}")
    summary_parts.append(f"**Localidades:** {metadata.get('total_localidades', 0)}")
    summary_parts.append(f"**Períodos cobertos:** {metadata.get('total_periodos', 0)}")

    # Lista de variáveis
    if metadata.get('variaveis'):
        summary_parts.append(f"")
        summary_parts.append(f"**Variáveis encontradas:**")
        for var in metadata['variaveis'][:10]:  # Limitar a 10
            summary_parts.append(f"- {var}")
        if len(metadata['variaveis']) > 10:
            summary_parts.append(f"- ... e mais {len(metadata['variaveis']) - 10}")

    # Períodos
    if metadata.get('periodos'):
        summary_parts.append(f"")
        summary_parts.append(f"**Períodos:** De {min(metadata['periodos'])} a {max(metadata['periodos'])}")

    # Estatísticas numéricas
    if 'estatisticas_numericas' in metadata:
        stats = metadata['estatisticas_numericas']
        summary_parts.append(f"")
        summary_parts.append(f"**Estatísticas dos valores numéricos:**")
        summary_parts.append(f"- Valores válidos: {stats['contagem']:,}")
        summary_parts.append(f"- Média: {stats['media']:,.2f}")
        summary_parts.append(f"- Mínimo: {stats['minimo']:,.2f}")
        summary_parts.append(f"- Máximo: {stats['maximo']:,.2f}")
        summary_parts.append(f"- Soma total: {stats['soma']:,.2f}")

    # Amostra dos dados
    if not df.empty:
        summary_parts.append(f"")
        summary_parts.append(f"**Amostra dos dados (primeiras 5 linhas):**")

        # Selecionar colunas principais para exibir
        display_cols = []
        for col in ['variavel', 'localidade', 'periodo', 'valor', 'unidade']:
            if col in df.columns:
                display_cols.append(col)

        if display_cols:
            sample_df = df[display_cols].head(5)

            # Criar tabela simples
            if len(display_cols) == 1:
                summary_parts.extend([f"- {val}" for val in sample_df.iloc[:, 0].tolist()])
            else:
                # Adicionar cabeçalho
                header = " | ".join([str(col).ljust(15)[:15] for col in display_cols])
                summary_parts.append(header)
                summary_parts.append("-" * len(header))

                # Adicionar linhas
                for _, row in sample_df.iterrows():
                    row_str = " | ".join([str(row[col]).ljust(15)[:15] for col in display_cols])
                    summary_parts.append(row_str)

    return "\n".join(summary_parts)