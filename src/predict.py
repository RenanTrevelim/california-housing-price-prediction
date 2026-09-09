from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ==================================================
# CAMINHOS
# ==================================================
ROOT = Path(__file__).resolve().parents[1]

CAMINHO_PIPELINE = ROOT / "models" / "pipeline_completo.pkl"


# ==================================================
# COLUNAS DE ENTRADA
# ==================================================
COLUNAS_MODELO = [
    "longitude",
    "latitude",
    "idade_mediana_imoveis",
    "total_comodos",
    "total_quartos",
    "populacao",
    "domicilios",
    "renda_media",
    "proximidade_oceano",
]


# ==================================================
# CARREGAMENTO DO PIPELINE
# ==================================================
@lru_cache(maxsize=1)
def carregar_pipeline():
    if not CAMINHO_PIPELINE.exists():
        raise FileNotFoundError(f"Pipeline não encontrado: {CAMINHO_PIPELINE}")

    return joblib.load(CAMINHO_PIPELINE)


# ==================================================
# VALIDAÇÃO
# ==================================================
def validar_dados(dados: pd.DataFrame) -> None:
    if dados.empty:
        raise ValueError("A base de dados está vazia.")

    colunas_ausentes = [coluna for coluna in COLUNAS_MODELO if coluna not in dados.columns]

    if colunas_ausentes:
        raise ValueError("Colunas ausentes: " + ", ".join(colunas_ausentes))


# ==================================================
# FEATURE ENGINEERING
# ==================================================
def criar_features(dados: pd.DataFrame) -> pd.DataFrame:
    resultado = dados.copy()

    resultado["comodos_por_domicilio"] = resultado["total_comodos"] / resultado["domicilios"].replace(0, np.nan)
    resultado["quartos_por_comodo"] = resultado["total_quartos"] / resultado["total_comodos"].replace(0, np.nan)

    return resultado


# ==================================================
# PREDIÇÃO
# ==================================================
def prever_valores(dados: pd.DataFrame) -> pd.Series:
    validar_dados(dados)

    pipeline = carregar_pipeline()

    entrada = dados[COLUNAS_MODELO].copy()
    entrada = criar_features(entrada)

    previsoes = pipeline.predict(entrada)

    return pd.Series(previsoes, index=dados.index, name="valor_estimado")