from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union


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


CollectionResult = Union[
    AssuntoResult,
    AgregadoResult,
]
