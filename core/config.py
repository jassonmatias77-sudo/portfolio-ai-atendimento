"""Configuração tipada do pipeline.

POR QUE PYDANTIC E NÃO UM DICT
    O esqueleto anterior fazia `config.get("model_version", "unknown")` — um
    erro de digitação na chave dava "unknown" em silêncio, e o pipeline corria
    com configuração errada.

    Com Pydantic, uma chave errada é erro na arranque, com mensagem clara.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ModeloConfig(BaseModel):
    """Qual modelo usar e como chamá-lo."""

    nome: str = Field(default="qwen2.5-coder:7b")
    base_url: str = Field(default="http://127.0.0.1:11434")
    temperatura: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=400, ge=1, le=8192)
    timeout_s: float = Field(default=120.0, gt=0)
    keep_alive: str = Field(
        default="30m",
        description="Quanto tempo o Ollama mantém o modelo na memória. "
                    "Sem isto, cada chamada recarrega o modelo (~2s).",
    )


class RagConfig(BaseModel):
    """Como indexar e recuperar documentos."""

    modelo_embedding: str = Field(default="nomic-embed-text:latest")
    pasta_docs: str = Field(default="docs")
    tamanho_chunk: int = Field(default=400, ge=50, le=4000)
    sobreposicao: int = Field(default=80, ge=0)
    top_k: int = Field(default=3, ge=1, le=20)
    score_minimo: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("sobreposicao")
    @classmethod
    def sobreposicao_menor_que_chunk(cls, v: int, info) -> int:
        """Sobreposição >= tamanho do chunk geraria loop infinito no chunking."""
        tamanho = info.data.get("tamanho_chunk", 400)
        if v >= tamanho:
            raise ValueError(
                "sobreposicao (%d) tem de ser menor que tamanho_chunk (%d)"
                % (v, tamanho))
        return v


class Config(BaseModel):
    """Configuração completa."""

    modelo: ModeloConfig = Field(default_factory=ModeloConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    idioma: str = Field(default="pt")
    prompt_sistema: str = Field(
        default=(
            "Responda em português do Brasil, de forma directa e técnica. "
            "Se o contexto fornecido não contiver a resposta, diga isso "
            "claramente em vez de inventar."
        ))


def carregar(caminho: str | Path = "config.json") -> Config:
    """Carrega e valida. Levanta erro claro se algo estiver errado."""
    import json

    p = Path(caminho)
    if not p.exists():
        return Config()          # defaults são válidos

    dados = json.loads(p.read_text(encoding="utf-8"))

    # compatibilidade: o config antigo tinha só {"model_version": "..."}
    if "model_version" in dados and "modelo" not in dados:
        dados = {"modelo": {"nome": "qwen2.5-coder:7b"},
                 "rag": {}, "idioma": "pt"}

    return Config(**dados)
