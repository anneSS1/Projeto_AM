# ################################################################
# PROJETO FINAL
#
# Universidade Federal de São Carlos (UFSCar)
# Departamento de Computação - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Máquina
# Prof. Tiago A. Almeida
#
# Nome: Anne Mari Suenaga Sakai e Felipe Jun Nishitani
# RA: 822304 e 822353
# ################################################################
#
# Arquivo com todas as funções e códigos referentes à análise exploratória

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.preprocessamento import gerar_dataset


# ---------------------------------------------------------
# 1. Leitura e informações gerais
# ---------------------------------------------------------
def carregar_e_resumir(base_path: str) -> pd.DataFrame:
    """Carrega o dataset consolidado e exibe informações básicas."""
    print("\n--- Carregando dataset ---")
    df = gerar_dataset(base_path)

    print("\n✅ Primeiras linhas:")
    display(df.head())

    print("\n📊 Informações gerais:")
    display(df.info())

    print("\n📈 Estatísticas descritivas:")
    display(df.describe())

    return df


# ---------------------------------------------------------
# 2. Medidas descritivas e contagens
# ---------------------------------------------------------
def medidas_descritivas(df: pd.DataFrame):
    """Exibe distribuição das classes e valores ausentes."""
    print("\n--- Medidas descritivas ---")

    print("\n📦 Distribuição das classes:")
    display(df["classe"].value_counts())

    plt.figure(figsize=(6,4))
    sns.countplot(x="classe", data=df, palette="Set2")
    plt.title("Distribuição das Classes")
    plt.show()

    print("\n🚨 Valores ausentes por coluna:")
    display(df.isnull().sum())


# ---------------------------------------------------------
# 3. Boxplots e histogramas
# ---------------------------------------------------------
def boxplots_e_histogramas(df: pd.DataFrame):
    """Gera boxplots e histogramas para os atributos numéricos."""
    num_cols = df.select_dtypes(include=np.number).columns

    print("\n--- Boxplots e Histogramas ---")

    # Boxplot geral
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df[num_cols], orient="h")
    plt.title("Boxplot Geral dos Atributos Numéricos")
    plt.show()

    # Boxplots por classe (limitado)
    for col in num_cols[:5]:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="classe", y=col, data=df, palette="coolwarm")
        plt.title(f"Boxplot de {col} por Classe")
        plt.show()

    # Histogramas
    for col in num_cols[:5]:
        plt.figure(figsize=(6,4))
        sns.histplot(data=df, x=col, hue="classe", kde=True)
        plt.title(f"Histograma de {col}")
        plt.show()


# ---------------------------------------------------------
# 4. Correlação entre atributos
# ---------------------------------------------------------
def correlacao_atributos(df: pd.DataFrame):
    """Plota a matriz de correlação entre os atributos numéricos."""
    print("\n--- Correlação entre atributos ---")
    num_df = df.select_dtypes(include=np.number)

    if num_df.empty:
        print("⚠️ Nenhum atributo numérico encontrado.")
        return

    corr = num_df.corr()

    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Matriz de Correlação entre Atributos Numéricos")
    plt.show()

    # Pega os pares de correlação (sem diagonais duplicadas)
    corr_pairs = corr.unstack().drop_duplicates()

    # Remove as correlações perfeitas (1.0)
    corr_pairs = corr_pairs[corr_pairs.abs() < 1]

    # Ordena por valor absoluto
    corr_pairs = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index)

    print("\n🔗 Maiores correlações encontradas:")
    display(corr_pairs.head(10))



# ---------------------------------------------------------
# 5. Pairplot (amostrado)
# ---------------------------------------------------------
def pairplot_amostrado(df: pd.DataFrame, n_amostras=300):
    """Gera um pairplot com amostra reduzida para evitar travamentos."""
    num_cols = df.select_dtypes(include=np.number).columns[:5]
    sample = df.sample(n=min(n_amostras, len(df)), random_state=42)

    print(f"\n--- Pairplot com {len(sample)} amostras e {len(num_cols)} atributos ---")
    sns.pairplot(sample[num_cols.tolist() + ["classe"]], hue="classe", height=2.5)
    plt.show()
