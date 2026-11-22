"""CLI do portfólio de engenharia de IA.

USO
    python main.py --pergunta "o que é RAG?"
    python main.py --indexar docs/
    python main.py --avaliar
    python main.py --diagnostico
"""
from __future__ import annotations

import argparse
import json
import sys

from core.config import carregar
from core.engine import Motor


def cmd_diagnostico(motor: Motor) -> int:
    d = motor.diagnostico()
    print("=== DIAGNÓSTICO ===")
    print("  Ollama     :", "OK" if d["ollama"] else "FORA")
    print("  modelos    :", len(d["modelos"]))
    print("  documentos :", d["documentos"])
    if d["faltando"]:
        print("\n  FALTA:")
        for f in d["faltando"]:
            print("    -", f)
        return 1
    print("\n  pronto para responder.")
    return 0


def cmd_indexar(motor: Motor, pasta: str) -> int:
    print("Indexando %s ..." % pasta)
    n = motor.indice.indexar_pasta(pasta)
    if n == 0:
        print("  nenhum documento encontrado (extensões: .md .txt .py .json)")
        return 1
    motor.indice.guardar("indice.json")
    print("  %d chunk(s) indexado(s) em %.2fs"
          % (n, motor.indice.tempo_indexacao_s))
    print("  guardado em indice.json")
    return 0


def cmd_pergunta(motor: Motor, pergunta: str, json_out: bool) -> int:
    # carrega índice guardado, se houver
    if motor.indice.carregar("indice.json"):
        pass

    r = motor.perguntar(pergunta)

    if json_out:
        print(json.dumps({
            "pergunta": pergunta,
            "resposta": r.texto,
            "fontes": r.fontes,
            "erro": r.erro,
            "medicao": r.medicao.para_dict(),
        }, ensure_ascii=False, indent=2))
        return 0 if r.ok else 1

    print("\n[pergunta] %s" % pergunta)
    if r.recuperados:
        melhor = max(x.score for x in r.recuperados)
        print("[rag]     %d documento(s), score máx %.2f, fontes: %s"
              % (len(r.recuperados), melhor, ", ".join(r.fontes) or "-"))
    else:
        print("[rag]     sem contexto recuperado")
    print("[llm]     %s" % motor.cfg.modelo.nome)

    if r.erro:
        print("\n[ERRO] %s" % r.erro)
        return 1

    print("\n[resposta] %s" % r.texto)
    print("\n[métricas] %s" % r.medicao.resumo())
    return 0


def cmd_avaliar(motor: Motor) -> int:
    # Carrega o índice guardado. Sem isto a avaliação corre SEM RAG e mede
    # apenas o modelo — foi o que aconteceu na primeira execução.
    n = motor.indice.carregar("indice.json")
    if n == 0:
        print("AVISO: indice.json ausente ou vazio. A avaliação correrá sem RAG.")
    from evaluation.bench import avaliar

    relatorio = avaliar(motor)
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    return 0 if relatorio.get("falhas", 1) == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Portfólio de engenharia de IA — pipeline com LLM real")
    ap.add_argument("--pergunta", "-p", help="pergunta a responder")
    ap.add_argument("--indexar", "-i", metavar="PASTA",
                    help="indexa uma pasta de documentos")
    ap.add_argument("--avaliar", "-a", action="store_true",
                    help="corre a avaliação quantitativa")
    ap.add_argument("--diagnostico", "-d", action="store_true",
                    help="verifica o que está pronto")
    ap.add_argument("--modelo", "-m", help="sobrepõe o modelo do config")
    ap.add_argument("--json", action="store_true", help="saída em JSON")
    ap.add_argument("--sem-rag", action="store_true",
                    help="responde sem consultar o índice")
    args = ap.parse_args(argv)

    cfg = carregar("config.json")
    if args.modelo:
        cfg.modelo.nome = args.modelo
    motor = Motor(cfg)

    if args.diagnostico:
        return cmd_diagnostico(motor)
    if args.indexar:
        return cmd_indexar(motor, args.indexar)
    if args.avaliar:
        return cmd_avaliar(motor)
    if args.pergunta:
        if args.sem_rag:
            motor.indice.documentos = []
        return cmd_pergunta(motor, args.pergunta, args.json)

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
