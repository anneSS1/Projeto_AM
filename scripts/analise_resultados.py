# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Anne Mari Suenaga Sakai
# RA: 822304
# 
# Nome: Felipe Jun Nishitani
# RA: 822353
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

sns.set(style="whitegrid")


# ======================================================================
# 1. Avaliação agregada: comparar modelos
# ======================================================================

def avaliar_modelos_dict(modelos: dict, X_val, y_val, names=None):
    """
    Avalia vários modelos em um conjunto de validação e retorna um DataFrame
    com métricas macro: accuracy, precision, recall e f1.
    """
    resultados = []
    ordem = names if names else modelos.keys()

    for nome in ordem:
        model = modelos[nome]
        y_pred = model.predict(X_val)

        resultados.append({
            "modelo": nome,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision_macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0)
        })

    return pd.DataFrame(resultados).set_index("modelo")


# ======================================================================
# 2. Relatórios detalhados por modelo
# ======================================================================

def mostrar_classification_reports(modelos: dict, X_val, y_val, names=None):
    """Imprime o classification_report para cada modelo."""
    ordem = names if names else modelos.keys()

    for nome in ordem:
        print(f"\n=== Classification Report: {nome} ===")
        y_pred = modelos[nome].predict(X_val)
        print(classification_report(y_val, y_pred, zero_division=0))


def tabela_por_classe(model, X_val, y_val, labels=None):
    """
    Retorna precisão, recall, f1 e suporte por classe para um modelo específico.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_pred = model.predict(X_val)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_pred, labels=labels, zero_division=0
    )

    classes = labels if labels is not None else np.unique(np.concatenate([y_val, y_pred]))

    return pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }, index=classes)


# ======================================================================
# 3. Visualizações: métricas e matrizes de confusão
# ======================================================================

def plot_metric_bars(df_metrics: pd.DataFrame, metrics=("accuracy", "f1_macro"), figsize=(8, 5)):
    """
    Plota barras comparando modelos nas métricas especificadas.
    """
    df_plot = df_metrics[list(metrics)]

    ax = df_plot.plot(kind="bar", figsize=figsize, rot=45)
    ax.set_title("Comparação de Métricas")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor")

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_model(model, X_val, y_val, labels=None, normalize=False, figsize=(6, 5)):
    """
    Plota a matriz de confusão para um único modelo.
    """
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, None]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão" + (" (Normalizada)" if normalize else ""))
    plt.show()


def comparar_modelos_plot_confusao(modelos: dict, X_val, y_val, labels=None,
                                   max_per_row=2, normalize=False):
    """
    Plota várias matrizes de confusão em grid para comparar modelos visualmente.
    """
    nomes = list(modelos.keys())
    n = len(nomes)
    cols = max_per_row
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 5, rows * 4))

    for i, nome in enumerate(nomes):
        model = modelos[nome]
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred, labels=labels)

        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, None]

        ax = plt.subplot(rows, cols, i + 1)
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                    cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(nome)

    plt.tight_layout()
    plt.show()


# ======================================================================
# 4. Curva de aprendizado
# ======================================================================

def plot_learning_curve_model(estimator, X, y, cv=5,
                              train_sizes=np.linspace(0.1, 1.0, 5),
                              figsize=(8, 5)):
    """
    Gera curva de aprendizado de um modelo (train x validation accuracy).
    Útil para detectar overfitting ou subfitting.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring="accuracy",
        shuffle=True,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=figsize)

    plt.plot(train_sizes, train_mean, marker="o", label="Treino")
    plt.plot(train_sizes, val_mean, marker="o", label="Validação")

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

    plt.title("Curva de Aprendizado")
    plt.xlabel("Tamanho do Conjunto de Treino")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ======================================================================
# 5. Teste estatístico (McNemar)
# ======================================================================

def teste_mcnemar(model1, model2, X_val, y_val):
    """
    Aplica o teste de McNemar para avaliar se dois classificadores
    têm diferença estatisticamente significativa.
    Retorna o p-valor.
    """
    y1 = model1.predict(X_val)
    y2 = model2.predict(X_val)

    tabela = [[0, 0], [0, 0]]

    for y_true, a, b in zip(y_val, y1, y2):
        tabela[int(a != y_true)][int(b != y_true)] += 1

    result = mcnemar(tabela, exact=False, correction=True)
    return result.pvalue