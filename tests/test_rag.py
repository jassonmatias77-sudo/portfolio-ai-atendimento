"""Testes do RAG: chunking, cosseno e recuperação."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rag import Indice, chunkar, cosseno


class ClienteFalso:
    """Finge embeddings para os testes não dependerem do Ollama.

    Embedding determinístico: o texto vira um vector de contagens de vogais.
    Não é semântico, mas é previsível e suficiente para testar o mecanismo.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.chamadas = 0

    def embeddings(self, modelo: str, texto: str) -> list[float]:
        self.chamadas += 1
        v = [0.0] * self.dim
        for i, c in enumerate(texto.lower()[:self.dim]):
            v[i % self.dim] += (ord(c) % 7) / 7.0
        return v


# ------------------------------------------------------------------ chunking
def test_texto_curto_nao_e_dividido():
    assert chunkar("texto curto", tamanho=400) == ["texto curto"]


def test_texto_longo_e_dividido():
    texto = "Frase de teste. " * 100
    pedacos = chunkar(texto, tamanho=200, sobreposicao=40)
    assert len(pedacos) > 1
    assert all(len(p) <= 260 for p in pedacos)


def test_sobreposicao_preserva_continuidade():
    texto = "abcdefghij" * 50
    pedacos = chunkar(texto, tamanho=100, sobreposicao=30)
    assert len(pedacos) >= 2
    # o fim de um aparece no início do seguinte
    assert pedacos[0][-20:] in pedacos[1] or pedacos[0][-10:] in pedacos[1]


def test_texto_vazio_devolve_lista_vazia():
    assert chunkar("") == []
    assert chunkar("   ") == []


def test_chunking_nao_entra_em_loop():
    """Sobreposição >= tamanho geraria loop infinito. Tem de terminar."""
    pedacos = chunkar("palavra " * 200, tamanho=50, sobreposicao=49)
    assert len(pedacos) < 500


# --------------------------------------------------------------------- cosseno
def test_cosseno_identico_e_1():
    v = [1.0, 2.0, 3.0]
    assert abs(cosseno(v, v) - 1.0) < 1e-9


def test_cosseno_ortogonal_e_0():
    assert abs(cosseno([1, 0], [0, 1])) < 1e-9


def test_cosseno_vazio_e_0_nao_explode():
    assert cosseno([], []) == 0.0
    assert cosseno([1, 2], []) == 0.0


def test_cosseno_tamanhos_diferentes_e_0():
    assert cosseno([1, 2, 3], [1, 2]) == 0.0


# ---------------------------------------------------------------- recuperação
def test_indexar_e_recuperar():
    idx = Indice(ClienteFalso(), "falso", tamanho_chunk=50, sobreposicao=10)
    n = idx.indexar_texto("o rato roeu a roupa do rei de roma", "teste")
    assert n >= 1
    assert len(idx.documentos) == n

    res = idx.recuperar("rato", top_k=2)
    assert len(res) <= 2
    assert res[0].score >= 0


def test_recuperar_sem_documentos_devolve_vazio():
    idx = Indice(ClienteFalso(), "falso")
    assert idx.recuperar("qualquer") == []


def test_indexar_pasta_conta_ficheiros(tmp_path):
    (tmp_path / "a.md").write_text("conteudo A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("conteudo B", encoding="utf-8")
    (tmp_path / "ignorar.log").write_text("x", encoding="utf-8")

    idx = Indice(ClienteFalso(), "falso", tamanho_chunk=100)
    n = idx.indexar_pasta(tmp_path)
    assert n == 2                       # .log fica de fora
    assert len(idx.documentos) == 2


def test_guardar_e_carregar(tmp_path):
    idx = Indice(ClienteFalso(), "falso", tamanho_chunk=100)
    idx.indexar_texto("documento de teste", "x")
    destino = tmp_path / "indice.json"
    idx.guardar(destino)

    novo = Indice(ClienteFalso(), "falso", tamanho_chunk=100)
    assert novo.carregar(destino) == len(idx.documentos)
    assert novo.documentos[0].texto == idx.documentos[0].texto
