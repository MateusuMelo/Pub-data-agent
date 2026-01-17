import csv
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from langchain.tools import tool

logger = logging.getLogger(__name__)


@tool
def criar_planilha_csv_dados_ibge(
        dados_coletados: Dict[str, Any],
        nome_arquivo: Optional[str] = None,
        diretorio: str = "output",
        incluir_metadados: bool = True
) -> str:
    """
    Cria uma planilha CSV com os dados coletados do IBGE.

    Args:
        dados_coletados: Dicionário com os dados do coletor
        nome_arquivo: Nome do arquivo CSV (se None, gera automaticamente)
        diretorio: Diretório para salvar o arquivo
        incluir_metadados: Se True, inclui uma aba/arquivo com metadados

    Returns:
        Caminho do arquivo CSV criado ou mensagem de erro
    """

    try:
        # Verificar se dados_coletados é um objeto Pydantic
        if hasattr(dados_coletados, "model_dump"):
            dados_coletados = dados_coletados.model_dump()

        # Extrair dados brutos
        raw_data = dados_coletados.get("raw_dados") or dados_coletados.get("collected_data")

        if not raw_data:
            return "❌ Nenhum dado disponível para criar planilha"

        # Se for lista com um item, extrair o item
        if isinstance(raw_data, list) and len(raw_data) == 1:
            raw_data = raw_data[0]

        # Gerar nome do arquivo se não fornecido
        if not nome_arquivo:
            metadata = dados_coletados.get("metadata", {})
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"ibge_dados_{metadata.get('agregado_id', 'unknown')}_{timestamp}.csv"

        # Garantir que tenha extensão .csv
        if not nome_arquivo.lower().endswith('.csv'):
            nome_arquivo += '.csv'

        # Criar diretório se não existir
        output_dir = Path(diretorio)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / nome_arquivo

        # Processar dados para CSV
        if isinstance(raw_data, dict):
            dados_para_csv = processar_dados_para_csv(raw_data, dados_coletados)
        elif isinstance(raw_data, list):
            dados_para_csv = []
            for item in raw_data:
                dados_para_csv.extend(processar_dados_para_csv(item, dados_coletados))
        else:
            return f"❌ Formato de dados não suportado: {type(raw_data)}"

        if not dados_para_csv:
            return "❌ Não foi possível extrair dados para o CSV"

        # Escrever CSV
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Obter cabeçalhos do primeiro item
            fieldnames = list(dados_para_csv[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in dados_para_csv:
                writer.writerow(row)

        # Criar arquivo de metadados se solicitado
        if incluir_metadados:
            metadata_path = output_dir / f"{nome_arquivo.replace('.csv', '_metadata.json')}"
            metadata_to_save = {
                "metadata": dados_coletados.get("metadata", {}),
                "source_used": dados_coletados.get("source_used", {}),
                "filters_applied": dados_coletados.get("filters_applied", {}),
                "collection_time": dados_coletados.get("collection_time"),
                "export_time": datetime.now().isoformat(),
                "total_rows": len(dados_para_csv)
            }

            with open(metadata_path, 'w', encoding='utf-8') as metafile:
                json.dump(metadata_to_save, metafile, ensure_ascii=False, indent=2)

        logger.info(f"✅ Planilha criada: {file_path} com {len(dados_para_csv)} linhas")

        return f"✅ Planilha CSV criada com sucesso: {file_path}\nTotal de registros: {len(dados_para_csv)}"

    except Exception as e:
        logger.error(f"❌ Erro ao criar planilha CSV: {e}")
        return f"❌ Erro ao criar planilha CSV: {str(e)}"


def processar_dados_para_csv(raw_data: Dict[str, Any], dados_coletados: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Processa dados brutos do IBGE para formato CSV.
    """
    rows = []

    # Extrair metadados
    metadata = dados_coletados.get("metadata", {})

    variavel_nome = raw_data.get("variavel", "N/A")
    variavel_id = raw_data.get("id", "N/A")
    unidade = raw_data.get("unidade", "N/A")

    # Processar cada resultado
    for resultado in raw_data.get("resultados", []):
        # Extrair classificações
        classificacoes_str = ""
        for classificacao in resultado.get("classificacoes", []):
            classif_id = classificacao.get("id", "")
            classif_nome = classificacao.get("nome", "")
            categoria = classificacao.get("categoria", {})

            if categoria:
                cats = ", ".join([f"{k}:{v}" for k, v in categoria.items()])
                classificacoes_str += f"{classif_nome}({cats}); "
            else:
                classificacoes_str += f"{classif_nome}; "

        # Processar cada série
        for serie in resultado.get("series", []):
            localidade = serie.get("localidade", {})
            serie_data = serie.get("serie", {})

            # Base row com informações comuns
            base_row = {
                "variavel_id": variavel_id,
                "variavel_nome": variavel_nome,
                "unidade": unidade,
                "classificacoes": classificacoes_str.rstrip("; "),
                "localidade_id": localidade.get("id", ""),
                "localidade_nome": localidade.get("nome", ""),
                "nivel_geografico_id": localidade.get("nivel", {}).get("id", ""),
                "nivel_geografico": localidade.get("nivel", {}).get("nome", ""),
                "agregado_id": metadata.get("agregado_id", ""),
                "periodo_id": metadata.get("periodo_id", ""),
                "territorio_id": metadata.get("territorio_id", ""),
                "classificacao_id": metadata.get("classificacao_id", "")
            }

            # Criar uma linha para cada período
            for periodo, valor in serie_data.items():
                row = base_row.copy()
                row["periodo"] = periodo
                row["valor"] = valor if valor is not None else ""
                rows.append(row)

    return rows


# Versão alternativa simplificada (uma linha por série)
@tool
def criar_planilha_csv_simplificada(
        dados_coletados: Dict[str, Any],
        nome_arquivo: Optional[str] = None
) -> str:
    """
    Cria uma planilha CSV simplificada com os dados do IBGE (uma linha por série).
    """

    try:
        if hasattr(dados_coletados, "model_dump"):
            dados_coletados = dados_coletados.model_dump()

        raw_data = dados_coletados.get("raw_dados") or dados_coletados.get("collected_data")

        if not raw_data:
            return "❌ Nenhum dado disponível"

        if isinstance(raw_data, list) and len(raw_data) == 1:
            raw_data = raw_data[0]

        # Gerar nome do arquivo
        if not nome_arquivo:
            metadata = dados_coletados.get("metadata", {})
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"ibge_simplificado_{metadata.get('agregado_id', 'data')}_{timestamp}.csv"

        if not nome_arquivo.lower().endswith('.csv'):
            nome_arquivo += '.csv'

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        file_path = output_dir / nome_arquivo

        # Processar dados
        rows = []
        metadata = dados_coletados.get("metadata", {})

        if isinstance(raw_data, dict):
            variavel = raw_data.get("variavel", "")
            unidade = raw_data.get("unidade", "")

            for resultado in raw_data.get("resultados", []):
                # Classificações como string
                classificacoes = []
                for classif in resultado.get("classificacoes", []):
                    cat = classif.get("categoria", {})
                    if cat:
                        classificacoes.append(f"{classif.get('nome')}:{','.join(f'{k}={v}' for k, v in cat.items())}")

                for serie in resultado.get("series", []):
                    localidade = serie.get("localidade", {})
                    serie_data = serie.get("serie", {})

                    # Converter série para string JSON
                    serie_json = json.dumps(serie_data, ensure_ascii=False)

                    row = {
                        "agregado_id": metadata.get("agregado_id", ""),
                        "variavel": variavel,
                        "unidade": unidade,
                        "classificacoes": "; ".join(classificacoes),
                        "localidade": localidade.get("nome", ""),
                        "nivel_geografico": localidade.get("nivel", {}).get("nome", ""),
                        "serie_temporal": serie_json,
                        "total_periodos": len(serie_data),
                        "periodos": ",".join(serie_data.keys()),
                        "valores": ",".join(str(v) if v is not None else "" for v in serie_data.values())
                    }
                    rows.append(row)

        # Escrever CSV
        if rows:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            return f"✅ Planilha simplificada criada: {file_path}\nRegistros: {len(rows)}"
        else:
            return "❌ Nenhum dado processado para CSV"

    except Exception as e:
        return f"❌ Erro: {str(e)}"