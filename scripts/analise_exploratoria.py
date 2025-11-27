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

    print("\nPrimeiras linhas do dataset:")
    exibir(df.head())

    print("\nInformações gerais:")
    df_info = df.info()          # info imprime sozinho
    print(df_info)

    print("\nEstatísticas descritivas:")
    exibir(df.describe(include="all"))

    return df


# ================================================================
# 2. Medidas descritivas
# ================================================================
def medidas_descritivas(df: pd.DataFrame):
    """Mostra distribuição das classes e valores ausentes."""
    print("\n--- Medidas Descritivas ---")

    if "classe" in df.columns:
        print("\nDistribuição das classes:")
        exibir(df["classe"].value_counts())

        plt.figure(figsize=(6, 4))
        # Correção do FutureWarning do Seaborn:
        #   Antes: sns.countplot(x="classe", data=df, palette="Set2")
        #   Seaborn mostrou warning dizendo que `palette` só deve ser usado quando houver `hue`, pois a paleta não é aplicada corretamente
        #   Solução: definir hue="classe"
        ax = sns.countplot(x="classe", data=df, hue="classe", palette="Set2")
        ax.get_legend().remove()  # Remove legenda duplicada
        plt.title("Distribuição das Classes")
        plt.show()
    else:
        print("Coluna 'classe' não encontrada no dataset.")

    print("\nValores ausentes por coluna:")
    exibir(df.isnull().sum())


# ================================================================
# 3. Boxplots e Histogramas
# ================================================================
def boxplots_e_histogramas(df: pd.DataFrame, top_n=6):
    """
    Gera boxplots e histogramas apenas para os atributos numéricos
    mais informativos, com gráficos organizados em grade.
    
    Critérios de seleção:
    - Top N atributos com maior variância
    - Top N atributos mais relevantes para a classe (ANOVA F-score)
    - Top N atributos com maior correlação absoluta entre si
    
    Isso evita gerar dezenas de gráficos irrelevantes e torna a EDA mais objetiva.
    """
    print("\n--- Boxplots e Histogramas (seleção inteligente) ---")

    # ============================
    # 1. Selecionar atributos numéricos
    # ============================
    num_df = df.select_dtypes(include=np.number)

    if num_df.empty:
        print("⚠️ Nenhuma coluna numérica encontrada.")
        return

    # ============================
    # 2. Ranking por variância
    # ============================
    variancias = num_df.var().sort_values(ascending=False)
    top_var = variancias.head(top_n).index.tolist()

    # ============================
    # 3. Ranking por relevância com a classe (ANOVA)
    # ============================
    top_fscore = []
    if "classe" in df.columns:
        try:
            from sklearn.feature_selection import f_classif
            f_vals, _ = f_classif(num_df, df["classe"])
            f_scores = pd.Series(f_vals, index=num_df.columns)
            top_fscore = f_scores.sort_values(ascending=False).head(top_n).index.tolist()
        except:
            pass

    # ============================
    # 4. Ranking por correlação
    # ============================
    corr_matrix = num_df.corr()
    corr_pairs = corr_matrix.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs < 1]           # tira diagonais
    corr_pairs = corr_pairs.sort_values(ascending=False)
    top_corr = list(dict.fromkeys([a for a, b in corr_pairs.index[:top_n]]))

    # ============================
    # 5. Seleção final dos atributos mais importantes
    # ============================
    atributos_escolhidos = list(dict.fromkeys(top_var + top_fscore + top_corr))
    atributos_escolhidos = atributos_escolhidos[:top_n]  # limitar ao desejado

    print(f"\n🧠 Atributos selecionados para visualização: {atributos_escolhidos}")

    # -------------------------------------------------------------------------
    # BOX-PLOTS (em grid 2 × N/2)
    # -------------------------------------------------------------------------
    n_cols = 2
    n_rows = int(np.ceil(len(atributos_escolhidos) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, atributos_escolhidos):
        sns.boxplot(x=df[col], ax=ax, color="skyblue")
        ax.set_title(f"Boxplot – {col}")

    # Apagar subplots vazios
    for i in range(len(atributos_escolhidos), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # HISTOGRAMAS (em grid)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, atributos_escolhidos):

        if "classe" in df.columns:
            # Histograma por classe
            for classe in df["classe"].unique():
                subset = df[df["classe"] == classe][col]
                ax.hist(subset, bins=25, alpha=0.5, label=str(classe))
            ax.legend()
        else:
            ax.hist(df[col], bins=25, alpha=0.7)

        ax.set_title(f"Histograma – {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequência")

    # Apagar subplots vazios
    for i in range(len(atributos_escolhidos), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
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
def pairplot_amostrado(df: pd.DataFrame, n_amostras=300):
    """
    Gera pairplot com amostra reduzida, selecionando automaticamente
    os atributos mais relevantes para motivar as features de interação.
    """

    print("\n--- Pairplot Amostrado (focado em interações fisiológicas) ---")

    # Lista final de atributos relevantes para justificar interações
    atributos_relevantes = []

    # 1) HR / ACC → índice de esforço cardíaco relativo ao movimento
    if "hr_mean" in df.columns and "acc_energy" in df.columns:
        atributos_relevantes += ["hr_mean", "acc_energy"]

    # 2) ACC × Temp Slope → movimento vs. resposta térmica periférica
    if "acc_energy" in df.columns and "temp_slope" in df.columns:
        atributos_relevantes += ["acc_energy", "temp_slope"]

    # 3) EDA × Temp Mean → ativação autonômica (suor + temperatura)
    if "eda_mean" in df.columns and "temp_mean" in df.columns:
        atributos_relevantes += ["eda_mean", "temp_mean"]

    # 4) HRV × EDA std → reatividade simpática
    if "hrv_rmssd" in df.columns and "eda_std" in df.columns:
        atributos_relevantes += ["hrv_rmssd", "eda_std"]

    # Remove duplicatas mantendo a ordem
    atributos_relevantes = list(dict.fromkeys(atributos_relevantes))

    if len(atributos_relevantes) == 0:
        print("Nenhum dos atributos necessários para interações foi encontrado.")
        return

    # Adiciona a classe se existir
    if "classe" in df.columns:
        cols = atributos_relevantes + ["classe"]
    else:
        cols = atributos_relevantes

    # Amostragem segura
    sample = df.sample(n=min(n_amostras, len(df)), random_state=42)
    print(f"Gerando pairplot com {len(sample)} amostras e {len(atributos_relevantes)} atributos...")

    sns.pairplot(
        sample[cols],
        hue="classe" if "classe" in df.columns else None,
        height=2.0,
        diag_kind="kde"
    )
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
