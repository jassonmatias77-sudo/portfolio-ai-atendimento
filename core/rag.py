"""RAG: indexação e recuperação de documentos.

POR QUE ISTO E NÃO UMA CAIXA NEGRA
    O valor de um portfólio está em mostrar o MECANISMO. Aqui está explícito:
    chunking, embedding, similaridade de cosseno, top-k. Sem framework que
    esconda o que acontece.

LIMITAÇÃO DECLARADA
    A busca é força bruta (comparar com todos os vectores). Acima de ~10 000
    documentos seria preciso HNSW. Está medido em evaluation/bench.py.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.client import ClienteOllama


@dataclass
class Documento:
    """Um pedaço de texto indexado."""

    texto: str
    fonte: str
    indice: int
    embedding: list[float] = field(default_factory=list)

    def para_json(self) -> dict:
        return asdict(self)


@dataclass
class Recuperado:
    """Um documento com o seu score."""

    documento: Documento
    score: float


def chunkar(texto: str, tamanho: int = 400, sobreposicao: int = 80) -> list[str]:
    """Divide o texto em pedaços com sobreposição.

    A sobreposição existe para não cortar uma ideia ao meio: se uma frase
    atravessa a fronteira, ela aparece inteira em pelo menos um chunk.

    Quebra preferencialmente em fim de frase, para o chunk não começar a meio.
    """
    if not texto or not texto.strip():
        return []

    texto = texto.strip()
    if len(texto) <= tamanho:
        return [texto]

    pedacos: list[str] = []
    inicio = 0
    n = len(texto)

    while inicio < n:
        fim = min(inicio + tamanho, n)

        # tenta terminar numa fronteira natural (desde 60% do chunk)
        if fim < n:
            minimo = inicio + int(tamanho * 0.6)
            for marca in ("\n\n", ". ", "! ", "? ", "\n", " "):
                pos = texto.rfind(marca, minimo, fim)
                if pos > inicio:
                    fim = pos + len(marca)
                    break

        pedaco = texto[inicio:fim].strip()
        if pedaco:
            pedacos.append(pedaco)

        avanca = fim - sobreposicao
        inicio = avanca if avanca > inicio else fim

    return pedacos


def cosseno(a: list[float], b: list[float]) -> float:
    """Similaridade de cosseno entre dois vectores.

    Devolve 0.0 se algum for vazio ou de tamanhos diferentes — melhor um
    resultado neutro do que uma excepção a meio da busca.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    p = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return p / (na * nb)


class Indice:
    """Índice de documentos com recuperação por similaridade."""

    def __init__(self, cliente: ClienteOllama, modelo_embedding: str,
                 tamanho_chunk: int = 400, sobreposicao: int = 80) -> None:
        self.cliente = cliente
        self.modelo = modelo_embedding
        self.tamanho = tamanho_chunk
        self.sobreposicao = sobreposicao
        self.documentos: list[Documento] = []
        self.tempo_indexacao_s: float = 0.0

    # ------------------------------------------------------------------ indexar
    def indexar_texto(self, texto: str, fonte: str = "inline") -> int:
        """Indexa um texto. Devolve quantos chunks foram criados."""
        pedacos = chunkar(texto, self.tamanho, self.sobreposicao)
        for i, p in enumerate(pedacos):
            emb = self.cliente.embeddings(self.modelo, p)
            self.documentos.append(
                Documento(texto=p, fonte=fonte, indice=i, embedding=emb))
        return len(pedacos)

    def indexar_ficheiro(self, caminho: str | Path) -> int:
        p = Path(caminho)
        if not p.exists() or not p.is_file():
            return 0
        try:
            conteudo = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return 0
        return self.indexar_texto(conteudo, fonte=p.name)

    def indexar_pasta(self, pasta: str | Path,
                      extensoes: tuple = (".md", ".txt", ".py", ".json")) -> int:
        base = Path(pasta)
        if not base.exists():
            return 0
        t0 = time.perf_counter()
        total = 0
        for f in sorted(base.rglob("*")):
            if f.is_file() and f.suffix.lower() in extensoes:
                total += self.indexar_ficheiro(f)
        self.tempo_indexacao_s = time.perf_counter() - t0
        return total

    # --------------------------------------------------------------- recuperar
    def recuperar(self, pergunta: str, top_k: int = 3,
                  score_minimo: float = 0.0) -> list[Recuperado]:
        """Top-k documentos mais similares à pergunta."""
        if not self.documentos:
            return []

        vetor = self.cliente.embeddings(self.modelo, pergunta)
        if not vetor:
            return []

        pontuados = [Recuperado(d, cosseno(vetor, d.embedding))
                     for d in self.documentos]
        pontuados.sort(key=lambda r: r.score, reverse=True)

        return [r for r in pontuados[:top_k] if r.score >= score_minimo]

    # ------------------------------------------------------------ persistência
    def guardar(self, caminho: str | Path) -> None:
        p = Path(caminho)
        p.parent.mkdir(parents=True, exist_ok=True)
        dados = [d.para_json() for d in self.documentos]
        p.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")

    def carregar(self, caminho: str | Path) -> int:
        p = Path(caminho)
        if not p.exists():
            return 0
        dados = json.loads(p.read_text(encoding="utf-8"))
        self.documentos = [Documento(**d) for d in dados]
        return len(self.documentos)
