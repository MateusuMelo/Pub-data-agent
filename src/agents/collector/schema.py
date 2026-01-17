from datetime import datetime
from typing import Optional, Dict, Any, List, Union

from pydantic import BaseModel, Field


class AssuntoResult(BaseModel):
    tipo: str = Field("assunto", description="Tipo do resultado")
    assunto_id: int = Field(description="ID do assunto IBGE")
    assunto_nome: str = Field(description="Nome do assunto IBGE")


class AgregadoResult(BaseModel):
    tipo: str = Field("agregado", description="Tipo do resultado")
    assunto_id: int = Field(description="ID do assunto usado")
    agregado_id: int = Field(description="ID do agregado IBGE")
    agregado_nome: str = Field(description="Nome do agregado")


class VariavelResult(BaseModel):
    tipo: str = Field("variavel", description="Tipo do resultado")
    variavel_id: int = Field(description="ID da variável IBGE")
    variavel_nome: str = Field(description="Nome da variável")


class PeriodoResult(BaseModel):
    tipo: str = Field("periodo", description="Tipo do resultado")
    periodo: str = Field(description="Período selecionado (ex: 2022)")


class CollectorCompleteResult(BaseModel):
    """Resultado completo da coleta para passar para o próximo agente."""
    success: bool
    collected_data: List[Any] = Field(default_factory=list)
    failed_variables: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_used: Dict[str, Any] = Field(default_factory=dict)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    collection_time: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Dados específicos para o próximo agente
    task: str
    parameters: Dict[str, Any]
    assunto_id: Optional[int] = None
    agregado_id: Optional[int] = None
    periodo_id: Optional[int] = None
    territorio_id: Optional[str] = None
    variavel_id: Optional[int] = None
    classificacao_id: Optional[int] = None
    raw_dados: Optional[List[Dict[str, Any]]] = None

    class Config:
        arbitrary_types_allowed = True


CollectionResult = Union[AssuntoResult, AgregadoResult, VariavelResult, PeriodoResult]
