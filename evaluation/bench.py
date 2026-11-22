"""Avaliação quantitativa do pipeline.

POR QUE AVALIAR
    "Funciona" não é uma medida. Este módulo mede latência por etapa e
    correspondência de conteúdo, para que uma regressão apareça como número.

O QUE MEDE
    latencia_media_s   tempo total por pergunta
    latencia_rag_s     tempo do RAG
    latencia_llm_s     tempo do modelo
    acertos            perguntas cuja resposta contém a palavra esperada
    falhas             perguntas sem resposta
    taxa_acerto        acertos / total
"""
from __future__ import annotations

import statistics
import time

# perguntas com a palavra que a resposta CERTA tem de conter.
# Deliberadamente simples: não mede semântica, mede se o pipeline entregou
# conteúdo relacionado.
CASOS = [
    ("O que e RAG?", ["contexto", "recuper", "retrieval", "documento"]),
    ("Para que serve um embedding?", ["vector", "vetor", "similaridade", "texto"]),
    ("O que e latencia?", ["tempo", "resposta", "demora"]),
]


def avaliar(motor) -> dict:
    """Corre os casos e devolve um relatório."""
    from core.telemetry import Medicao

    resultados = []
    t0 = time.perf_counter()

    for pergunta, esperados in CASOS:
        r = motor.perguntar(pergunta)
        texto = (r.texto or "").lower()
        acertou = any(e.lower() in texto for e in esperados)
        resultados.append({
            "pergunta": pergunta,
            "acertou": acertou,
            "erro": r.erro,
            "latencia_s": round(r.medicao.total_s, 3),
            "rag_s": round(r.medicao.tempo_de("rag"), 3),
            "llm_s": round(r.medicao.tempo_de("llm"), 3),
            "tamanho_resposta": len(r.texto or ""),
        })

    total = len(resultados)
    acertos = sum(1 for x in resultados if x["acertou"])
    falhas = sum(1 for x in resultados if x["erro"])
    lat = [x["latencia_s"] for x in resultados if x["latencia_s"] > 0]

    return {
        "modelo": motor.cfg.modelo.nome,
        "documentos_indexados": len(motor.indice.documentos),
        "total_perguntas": total,
        "acertos": acertos,
        "falhas": falhas,
        "taxa_acerto": round(acertos / total, 2) if total else 0.0,
        "latencia_media_s": round(statistics.mean(lat), 3) if lat else 0.0,
        "latencia_max_s": round(max(lat), 3) if lat else 0.0,
        "duracao_total_s": round(time.perf_counter() - t0, 2),
        "detalhe": resultados,
    }
