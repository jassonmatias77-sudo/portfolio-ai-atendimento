"""Testes da configuração tipada."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, carregar


def test_defaults_sao_validos():
    c = Config()
    assert c.modelo.nome
    assert c.rag.tamanho_chunk > c.rag.sobreposicao


def test_config_inexistente_usa_defaults(tmp_path):
    c = carregar(tmp_path / "nao_existe.json")
    assert c.modelo.nome            # não levanta, usa defaults


def test_config_invalido_levanta_erro(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"modelo": {"temperatura": 5.0}}), encoding="utf-8")
    with pytest.raises(Exception):
        carregar(p)                 # temperatura > 2.0


def test_sobreposicao_maior_que_chunk_e_recusada(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "rag": {"tamanho_chunk": 100, "sobreposicao": 200}
    }), encoding="utf-8")
    with pytest.raises(Exception):
        carregar(p)


def test_config_antigo_e_migrado(tmp_path):
    """O esqueleto anterior tinha {"model_version": "..."} apenas."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"model_version": "v1.0-default"}), encoding="utf-8")
    c = carregar(p)
    assert c.modelo.nome            # migrou sem rebentar


def test_keep_alive_tem_default():
    assert Config().modelo.keep_alive
