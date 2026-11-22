"""API HTTP do pipeline.

POR QUE UMA API ALÉM DA CLI
    Permite que outro sistema use isto como serviço — que é como a IA é
    consumida em produção. A CLI serve para demonstração, a API para integração.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.config import carregar
from core.engine import Motor

app = FastAPI(title="Portfólio IA", version="2.0")
_motor: Motor | None = None


def motor() -> Motor:
    global _motor
    if _motor is None:
        _motor = Motor(carregar("config.json"))
        _motor.indice.carregar("indice.json")
    return _motor


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=1, max_length=2000)
    usar_rag: bool = True


class RespostaAPI(BaseModel):
    resposta: str
    fontes: list[str]
    erro: str
    metricas: dict


@app.get("/health")
def health() -> dict:
    """Estado sem tocar no modelo — rápido."""
    m = motor()
    return {"status": "ok",
            "documentos": len(m.indice.documentos),
            "modelo": m.cfg.modelo.nome}


@app.get("/diagnostico")
def diagnostico() -> dict:
    return motor().diagnostico()


@app.post("/perguntar", response_model=RespostaAPI)
def perguntar(p: Pergunta) -> RespostaAPI:
    m = motor()
    r = m.perguntar(p.pergunta, usar_rag=p.usar_rag)
    if r.erro and not r.texto:
        raise HTTPException(status_code=503, detail=r.erro)
    return RespostaAPI(resposta=r.texto, fontes=r.fontes, erro=r.erro,
                       metricas=r.medicao.para_dict())
