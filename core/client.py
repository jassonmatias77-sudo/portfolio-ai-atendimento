"""Cliente Ollama com retry e telemetria.

POR QUE UM CLIENTE PRÓPRIO E NÃO CHAMADAS SOLTAS
    Cada chamada precisa de: timeout, retry, medição de tempo e tratamento de
    erro específico. Espalhar isso pelo código garante que uma delas vai
    esquecer o timeout — e o pipeline fica pendurado para sempre.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests


class OllamaIndisponivel(RuntimeError):
    """O servidor não respondeu. Erro de infraestrutura, não de lógica."""


class ModeloNaoEncontrado(RuntimeError):
    """O modelo não existe no Ollama. Erro de configuração."""


@dataclass
class Resultado:
    """O que uma chamada devolve, com o que é preciso medir."""

    texto: str
    modelo: str
    latencia_s: float
    tokens: int = 0
    tentativas: int = 1
    meta: dict = field(default_factory=dict)


class ClienteOllama:
    """Cliente mínimo, com retry e medição."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434",
                 timeout_s: float = 120.0, max_tentativas: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tentativas = max(1, max_tentativas)

    # ------------------------------------------------------------------ saúde
    def disponivel(self, timeout: float = 8.0) -> bool:
        """O servidor responde? Nunca levanta — devolve bool."""
        try:
            r = requests.get(self.base_url + "/api/tags", timeout=timeout)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def modelos(self) -> list[str]:
        """Nomes dos modelos instalados."""
        try:
            r = requests.get(self.base_url + "/api/tags", timeout=15)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:  # noqa: BLE001
            raise OllamaIndisponivel("Ollama não respondeu: %s" % e) from e

    # ------------------------------------------------------------------ chat
    def gerar(self, modelo: str, mensagens: list[dict],
              temperatura: float = 0.2, max_tokens: int = 400,
              keep_alive: str = "30m") -> Resultado:
        """Chama /api/chat com retry.

        Levanta OllamaIndisponivel se o servidor não responde, e
        ModeloNaoEncontrado se o modelo não existe. A distinção importa:
        a primeira é temporária, a segunda exige acção.
        """
        payload = {
            "model": modelo,
            "messages": mensagens,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": temperatura, "num_predict": max_tokens},
        }

        ultimo_erro = None
        for tentativa in range(1, self.max_tentativas + 1):
            t0 = time.perf_counter()
            try:
                r = requests.post(self.base_url + "/api/chat", json=payload,
                                  timeout=self.timeout_s)
            except requests.exceptions.Timeout as e:
                ultimo_erro = "timeout após %.0fs" % self.timeout_s
                if tentativa < self.max_tentativas:
                    time.sleep(1.5 * tentativa)
                    continue
                raise OllamaIndisponivel(ultimo_erro) from e
            except Exception as e:  # noqa: BLE001
                ultimo_erro = "%s: %s" % (type(e).__name__, e)
                if tentativa < self.max_tentativas:
                    time.sleep(1.5 * tentativa)
                    continue
                raise OllamaIndisponivel(ultimo_erro) from e

            dur = time.perf_counter() - t0

            if r.status_code == 404:
                # modelo inexistente: retry não ajuda
                raise ModeloNaoEncontrado(
                    "modelo '%s' não existe no Ollama. Instale com: "
                    "ollama pull %s" % (modelo, modelo))

            if r.status_code != 200:
                ultimo_erro = "HTTP %s: %s" % (r.status_code, r.text[:150])
                if tentativa < self.max_tentativas:
                    time.sleep(1.5 * tentativa)
                    continue
                raise OllamaIndisponivel(ultimo_erro)

            d = r.json()
            msg = d.get("message", {}) or {}
            return Resultado(
                texto=(msg.get("content") or "").strip(),
                modelo=modelo,
                latencia_s=dur,
                tokens=int(d.get("eval_count") or 0),
                tentativas=tentativa,
                meta={"done_reason": d.get("done_reason", "")},
            )

        raise OllamaIndisponivel(ultimo_erro or "falha desconhecida")

    # -------------------------------------------------------------- embeddings
    def embeddings(self, modelo: str, texto: str) -> list[float]:
        """Vector de embedding. 768 dimensões com nomic-embed-text."""
        try:
            r = requests.post(self.base_url + "/api/embeddings",
                              json={"model": modelo, "prompt": texto},
                              timeout=self.timeout_s)
        except Exception as e:  # noqa: BLE001
            raise OllamaIndisponivel("embeddings: %s" % e) from e

        if r.status_code == 404:
            raise ModeloNaoEncontrado(
                "modelo de embedding '%s' não existe. Instale com: "
                "ollama pull %s" % (modelo, modelo))
        if r.status_code != 200:
            raise OllamaIndisponivel("embeddings HTTP %s" % r.status_code)

        return r.json().get("embedding", []) or []
