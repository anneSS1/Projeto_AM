# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Anne Mari Suenaga Sakai e Felipe Jun Nishitani
# RA: 822304 e 822353
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sns.set(style="whitegrid")


def avaliar_modelos_dict(modelos: dict, X_val, y_val, names=None):
    """
    Avalia um dicionário de modelos em (X_val, y_val).
    modelos: dict(nome -> modelo treinado)
    names: lista opcional para reordenar/filtrar
    Retorna: DataFrame com metrics (accuracy, precision_macro, recall_macro, f1_macro)
    """
    resultados = []
    ordem = names if names is not None else list(modelos.keys())

    for nome in ordem:
        model = modelos[nome]
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        resultados.append({
            "modelo": nome,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1
        })

    df_res = pd.DataFrame(resultados).set_index("modelo")
    return df_res


def mostrar_classification_reports(modelos: dict, X_val, y_val, names=None):
    """Imprime classification_report para cada modelo."""
    ordem = names if names is not None else list(modelos.keys())
    for nome in ordem:
        print(f"\n=== Classification report: {nome} ===")
        model = modelos[nome]
        y_pred = model.predict(X_val)
        print(classification_report(y_val, y_pred, zero_division=0))


def plot_metric_bars(df_metrics: pd.DataFrame, metrics=("accuracy", "f1_macro"), figsize=(8, 5)):
    """
    Plota barras comparativas para as métricas especificadas (tupla de colnames do df_metrics).
    df_metrics: DataFrame index=modelo, colunas métricas
    """
    df_plot = df_metrics[list(metrics)]
    ax = df_plot.plot(kind="bar", figsize=figsize, rot=45)
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    ax.set_title("Comparação de Métricas por Modelo")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, rotation=0)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_model(model, X_val, y_val, labels=None, cmap="Blues", normalize=False, figsize=(6,5)):
    """
    Plota matriz de confusão para um modelo.
    normalize: se True, mostra proporções por linha (recall por classe)
    """
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão" + (" (normalizada)" if normalize else ""))
    plt.show()


def tabela_por_classe(model, X_val, y_val, labels=None):
    """
    Retorna um DataFrame com precision/recall/f1 por classe para um modelo.
    """
    from sklearn.metrics import precision_recall_fscore_support
    y_pred = model.predict(X_val)
    precision, recall, f1, support = precision_recall_fscore_support(y_val, y_pred, zero_division=0, labels=labels)
    classes = labels if labels is not None else np.unique(np.concatenate([y_val, y_pred]))
    df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }, index=classes)
    return df


def salvar_resumo_csv(df_metrics: pd.DataFrame, path="resultados_summary.csv"):
    """
    Salva o DataFrame de métricas em CSV.
    """
    df_metrics.to_csv(path, float_format="%.6f")
    print(f"✅ Resumo salvo em: {path}")


def comparar_modelos_plot_confusao(modelos: dict, X_val, y_val, labels=None, max_per_row=2, normalize=False):
    """
    Plota matrizes de confusão para todos modelos do dicionário, em grid.
    """
    nomes = list(modelos.keys())
    n = len(nomes)
    cols = max_per_row
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols*5, rows*4))
    for i, nome in enumerate(nomes):
        model = modelos[nome]
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred, labels=labels)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        ax = plt.subplot(rows, cols, i+1)
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(nome)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
    plt.tight_layout()
    plt.show()


