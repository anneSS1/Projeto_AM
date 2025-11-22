# ################################################################
# PROJETO FINAL
#
# Universidade Federal de São Carlos (UFSCar)
# Departamento de Computação - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Máquina
# Prof. Tiago A. Almeida
#
# Nome: Anne Mari Suenaga Sakai
# RA: 822304
# 
# Nome: Felipe Jun Nishitani
# RA: 822353
# ################################################################
#
# Arquivo com todas as funções e códigos referentes à análise exploratória

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.preprocessamento import gerar_dataset

# ================================================================
# Função utilitária para exibir no Notebook OU no terminal
# ================================================================
def exibir(obj):
    """Usa display() se existir (notebook), senão usa print()."""
    try:
        display(obj)
    except:
        print(obj)


# ================================================================
# 1. Carregar dataset + resumo geral
# ================================================================
def carregar_e_resumir(base_path: str) -> pd.DataFrame:
    """
    Carrega o dataset consolidado gerado pelo preprocessamento e exibe:
        - head()
        - info()
        - describe()
    """
    print("\n--- Carregando dataset ---")
    df = gerar_dataset(base_path)

    print("\n✅ Primeiras linhas do dataset:")
    exibir(df.head())

    print("\n📊 Informações gerais:")
    df_info = df.info()          # info imprime sozinho
    print(df_info)

    print("\n📈 Estatísticas descritivas:")
    exibir(df.describe(include="all"))

    return df


# ================================================================
# 2. Medidas descritivas
# ================================================================
def medidas_descritivas(df: pd.DataFrame):
    """Mostra distribuição das classes e valores ausentes."""
    print("\n--- Medidas Descritivas ---")

    if "classe" in df.columns:
        print("\n📦 Distribuição das classes:")
        exibir(df["classe"].value_counts())

        plt.figure(figsize=(6, 4))
        sns.countplot(x="classe", data=df, palette="Set2")
        plt.title("Distribuição das Classes")
        plt.show()
    else:
        print("⚠️ Coluna 'classe' não encontrada no dataset.")

    print("\n🚨 Valores ausentes por coluna:")
    exibir(df.isnull().sum())


# ================================================================
# 3. Boxplots e Histogramas
# ================================================================
def boxplots_e_histogramas(df: pd.DataFrame, max_cols=6):
    """
    Gera boxplots e histogramas para atributos numéricos (versão robusta sem KDE).
    """
    print("\n--- Boxplots e Histogramas ---")

    # Selecionar apenas colunas numéricas
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) == 0:
        print("⚠️ Nenhuma coluna numérica encontrada para visualização.")
        return

    num_cols = num_cols[:max_cols]

    # Boxplot geral
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df[num_cols], orient="h")
    plt.title("Boxplot Geral dos Atributos Numéricos")
    plt.show()

    # Boxplot por classe
    if "classe" in df.columns:
        for col in num_cols:
            if df[col].nunique() < 2:
                continue
            plt.figure(figsize=(6, 4))
            sns.boxplot(x="classe", y=col, data=df, palette="Spectral")
            plt.title(f"Boxplot – {col} por Classe")
            plt.show()

    # Histogramas 
    for col in num_cols[:5]:
        if df[col].nunique() < 2:
            print(f"⚠️ Coluna {col} ignorada (variância zero).")
            continue

        plt.figure(figsize=(6,4))

        if "classe" in df.columns:
            # Plot separado por classe
            for classe in df["classe"].unique():
                subset = df[df["classe"] == classe][col]
                plt.hist(subset, bins=30, alpha=0.5, label=str(classe))
            plt.legend()
        else:
            plt.hist(df[col], bins=30)

        plt.title(f"Histograma de {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")
        plt.show()



# ================================================================
# 4. Correlação entre atributos
# ================================================================
def correlacao_atributos(df: pd.DataFrame):
    """Plota a matriz de correlação para todos os atributos numéricos."""
    print("\n--- Matriz de Correlação ---")

    num_df = df.select_dtypes(include=np.number)

    if num_df.empty:
        print("⚠️ Nenhuma coluna numérica encontrada.")
        return

    corr = num_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Matriz de Correlação entre Atributos Numéricos")
    plt.show()

    # Correlações ordenadas
    corr_pairs = corr.unstack().drop_duplicates()
    corr_pairs = corr_pairs[corr_pairs.abs() < 1]  # remove diagonais
    corr_pairs = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index)

    print("\n🔗 Maiores correlações:")
    exibir(corr_pairs.head(10))


# ================================================================
# 5. Pairplot com amostra
# ================================================================
def pairplot_amostrado(df: pd.DataFrame, n_amostras=300, n_features=5):
    """
    Gera pairplot com amostra reduzida.
    Evita travamentos em datasets grandes.
    """
    print("\n--- Pairplot Amostrado ---")

    num_cols = df.select_dtypes(include=np.number).columns[:n_features].tolist()

    if "classe" in df.columns:
        cols = num_cols + ["classe"]
    else:
        cols = num_cols

    # Amostra segura
    sample = df.sample(n=min(n_amostras, len(df)), random_state=42)

    print(f"Gerando pairplot com {len(sample)} amostras e {len(num_cols)} atributos...")

    sns.pairplot(sample[cols], hue="classe" if "classe" in df.columns else None,
                 height=2.0, diag_kind="kde")
    plt.show()



# ################################################################
# 6. Análise de séries temporais
# ################################################################
def plot_series_temporais(df, colunas):
    print("\n--- Séries Temporais ---")

    for col in colunas:
        if col not in df.columns:
            print(f"⚠️ Coluna {col} não encontrada.")
            continue
        
        plt.figure(figsize=(12,4))
        plt.plot(df[col].values)
        plt.title(f"Série Temporal – {col}")
        plt.xlabel("Tempo (amostra)")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()


################################################################
# 7. Scatter plot duplo
################################################################
def scatter_duplo(df, x, y):
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=x, y=y, hue="classe", alpha=0.6)
    plt.title(f"{x} vs {y}")
    plt.show()
