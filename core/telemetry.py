"""Telemetria: medir cada etapa do pipeline.

POR QUE ISTO EXISTE
    A primeira versão media só o total (2,1 s). Sem saber ONDE o tempo ia,
    assumi que o LLM era o gargalo. Medindo por etapa, descobri que 0,3 s era
    o Ollama a carregar o modelo — resolvível com keep_alive.

    Quem não mede, optimiza o sítio errado.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Etapa:
    nome: str
    duracao_s: float
    detalhe: str = ""


@dataclass
class Medicao:
    """O que aconteceu num turno."""

    etapas: list[Etapa] = field(default_factory=list)
    inicio: float = field(default_factory=time.perf_counter)

    def registar(self, nome: str, duracao_s: float, detalhe: str = "") -> None:
        self.etapas.append(Etapa(nome, duracao_s, detalhe))

    @property
    def total_s(self) -> float:
        return time.perf_counter() - self.inicio

    def tempo_de(self, nome: str) -> float:
        for e in self.etapas:
            if e.nome == nome:
                return e.duracao_s
        return 0.0

    def para_dict(self) -> dict:
        return {
            "total_s": round(self.total_s, 3),
            "etapas": [{"nome": e.nome,
                        "s": round(e.duracao_s, 3),
                        "detalhe": e.detalhe} for e in self.etapas],
        }

    def resumo(self) -> str:
        partes = ["total %.2fs" % self.total_s]
        for e in self.etapas:
            partes.append("%s %.2fs" % (e.nome, e.duracao_s))
        return " | ".join(partes)


@contextmanager
def medir(medicao: Medicao, nome: str, detalhe: str = ""):
    """Mede um bloco de código.

    Uso:
        with medir(m, "rag"):
            docs = indice.recuperar(pergunta)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        medicao.registar(nome, time.perf_counter() - t0, detalhe)
