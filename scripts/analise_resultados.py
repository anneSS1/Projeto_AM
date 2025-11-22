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
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance

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
    df_metrics deve ter índice = nome do modelo e colunas com métricas.
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
        # evitar divisão por zero
        row_sums = cm.sum(axis=1)[:, None]
        with np.errstate(all='ignore'):
            cm = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
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
            row_sums = cm.sum(axis=1)[:, None]
            with np.errstate(all='ignore'):
                cm = np.divide(cm.astype(float), row_sums, where=row_sums != 0)

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

    # construir tabela 2x2: [ (acerto/acerto), (acerto/erro); (erro/acerto), (erro/erro) ]
    # a convenção usada abaixo conta erros
    tabela = [[0, 0], [0, 0]]

    for y_true, a, b in zip(y_val, y1, y2):
        tabela[int(a != y_true)][int(b != y_true)] += 1

    result = mcnemar(tabela, exact=False, correction=True)
    return result.pvalue


# ======================================================================
# 6. Curvas Precision-Recall e ROC multiclass
# ======================================================================

def plot_precision_recall_multiclass(models, X, y, label_encoder=None, figsize=(8, 6)):
    """
    Plota curvas Precision-Recall para cada classe (multiclass).
    models: dict[name] = model OR model (single)
    Se for dict, plota para cada modelo (uma figura por modelo).
    label_encoder: opcional, para obter nomes das classes; caso contrário usa labels numéricos.
    """
    # helper interno: processa um par (name, model)
    def _plot_for(name, model):
        y_true = y
        classes = np.unique(y_true)
        n_classes = len(classes)
        # binarizar rótulos
        y_bin = label_binarize(y_true, classes=classes)

        # tentar obter probabilidades
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)
        else:
            # fallback: determinístico (one-hot da predição)
            preds = model.predict(X)
            y_score = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                y_score[i, p] = 1.0

        plt.figure(figsize=figsize)
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
            ap = average_precision_score(y_bin[:, i], y_score[:, i])
            plt.plot(recall, precision, lw=2, label=f"classe {cls} (AP={ap:.2f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve — {name}")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    if isinstance(models, dict):
        for name, model in models.items():
            _plot_for(name, model)
    else:
        _plot_for("model", models)


def plot_roc_multiclass(models, X, y, label_encoder=None, figsize=(8, 6)):
    """
    Plota ROC multiclass (curvas por classe e AUC macro) para cada modelo.
    """
    def _plot_for(name, model):
        y_true = y
        classes = np.unique(y_true)
        n_classes = len(classes)
        y_bin = label_binarize(y_true, classes=classes)

        # obter probabilidades / scores
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)
        else:
            # usar predição dura como fallback
            preds = model.predict(X)
            y_score = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                y_score[i, p] = 1.0

        # curvas por classe
        plt.figure(figsize=figsize)
        aucs = []
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            auc_val = auc(fpr, tpr)
            aucs.append(auc_val)
            plt.plot(fpr, tpr, label=f"Classe {cls} (AUC={auc_val:.2f})")

        # micro-average (opcional)
        # construir fpr/tpr micro
        try:
            fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
            auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro (AUC={auc_micro:.2f})")
        except Exception:
            pass

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Multiclass — {name}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    if isinstance(models, dict):
        for name, model in models.items():
            _plot_for(name, model)
    else:
        _plot_for("model", models)


# ======================================================================
# 7. Importância de features
# ======================================================================

def importancia_features_modelo(model, X, y, feature_names=None, top_k=20, random_state=42):
    """
    Tenta estimar a importância das features para um modelo.
    Estratégias:
      - Se model tem coef_ (ex: LogisticRegression), usa coef absoluto médio por classe.
      - Caso contrário, usa permutation importance (scikit-learn).
    Retorna DataFrame ordenado por importância.
    """
    # tentar coef_
    try:
        coef = getattr(model, "coef_", None)
        if coef is not None:
            # coef shape (n_classes, n_features) ou (n_features,)
            coef_arr = np.array(coef)
            if coef_arr.ndim == 1:
                imp = np.abs(coef_arr)
            else:
                # média absoluta entre classes
                imp = np.mean(np.abs(coef_arr), axis=0)
            names = feature_names if feature_names is not None else [f"f{i}" for i in range(len(imp))]
            df_imp = pd.DataFrame({"feature": names, "importance": imp})
            df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
            return df_imp.head(top_k)
    except Exception:
        pass

    # fallback: permutation importance
    try:
        r = permutation_importance(model, X, y, n_repeats=20, random_state=random_state, n_jobs=-1)
        imp = r.importances_mean
        names = feature_names if feature_names is not None else [f"f{i}" for i in range(len(imp))]
        df_imp = pd.DataFrame({"feature": names, "importance": imp})
        df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp.head(top_k)
    except Exception as e:
        print("Falha ao calcular importância:", e)
        return pd.DataFrame({"feature": [], "importance": []})


def plot_importancia(df_imp, figsize=(8, 6), top_k=15):
    """
    Plota um barplot de importância (DataFrame com colunas 'feature' e 'importance').
    """
    if df_imp.empty:
        print("Nenhuma importância para mostrar.")
        return

    df_plot = df_imp.head(top_k).copy()
    plt.figure(figsize=figsize)
    sns.barplot(x="importance", y="feature", data=df_plot, orient="h")
    plt.title("Importância das Features")
    plt.tight_layout()
    plt.show()


# ======================================================================
# 8. Análise de erros / Confusões detalhadas
# ======================================================================

def analise_erros_por_classe(model, X_val, y_val, feature_df=None, labels=None):
    """
    Analisa exemplos errados por classe: retorna DataFrame com contagem de falsos positivos,
    falsos negativos e exemplos confusos. Se feature_df for fornecido (DataFrame original das features),
    também calcula médias das features onde ocorrem erros para entender padrões.
    """
    y_pred = model.predict(X_val)
    classes = labels if labels is not None else np.unique(np.concatenate([y_val, y_pred]))

    rows = []
    for cls in classes:
        # falsos negativos: verdadeiros cls, predito != cls
        fn_idx = np.where((y_val == cls) & (y_pred != cls))[0]
        # falsos positivos: predito cls, verdade != cls
        fp_idx = np.where((y_val != cls) & (y_pred == cls))[0]
        # verdadeiros corretos
        tp_idx = np.where((y_val == cls) & (y_pred == cls))[0]

        row = {
            "classe": cls,
            "FN_count": len(fn_idx),
            "FP_count": len(fp_idx),
            "TP_count": len(tp_idx),
            "FN_percent": len(fn_idx) / max(1, len(y_val)),
            "FP_percent": len(fp_idx) / max(1, len(y_val))
        }

        # se feature_df fornecido, calcular média das features nos erros
        if feature_df is not None:
            try:
                fn_mean = feature_df.iloc[fn_idx].mean().to_dict() if len(fn_idx) > 0 else {}
                fp_mean = feature_df.iloc[fp_idx].mean().to_dict() if len(fp_idx) > 0 else {}
                row["FN_feature_mean"] = fn_mean
                row["FP_feature_mean"] = fp_mean
            except Exception:
                row["FN_feature_mean"] = {}
                row["FP_feature_mean"] = {}

        rows.append(row)

    return pd.DataFrame(rows)
