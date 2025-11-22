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

# Arquivo com todas as funções e códigos referentes aos experimentos

import os
import pandas as pd
import numpy as np

# Modelos
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# ===================================================================
# 1. Treinamento dos modelos (BÁSICOS / BASELINES)
# ===================================================================

def treinar_knn(X_train, y_train, k=7):
    """
    Treina um modelo KNN simples (baseline).
    """
    print(f"\n--- Treinando KNN (k={k}) ---")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def treinar_naive_bayes(X_train, y_train):
    """
    Treina Naive Bayes (baseline).
    """
    print("\n--- Treinando Naive Bayes (Gaussian) ---")
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def treinar_regressao_logistica(X_train, y_train):
    """
    Regressão logística (baseline forte).
    """
    print("\n--- Treinando Regressão Logística ---")
    model = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def treinar_rede_neural(X_train, y_train):
    """
    Rede neural MLP simples (baseline).
    """
    print("\n--- Treinando Rede Neural (MLP) ---")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def treinar_svm(X_train, y_train):
    """
    SVM simples (baseline).
    """
    print("\n--- Treinando SVM (RBF) ---")
    model = SVC(
        kernel='rbf',
        probability=True,
        C=1.0,
        gamma='scale',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model



# ===================================================================
# 2. AVALIAÇÃO VIA VALIDAÇÃO CRUZADA
# ===================================================================

def avaliar_com_crossval(model, X, y, cv=5, scoring="accuracy"):
    """
    Executa validação cruzada (k-fold).
    Retorna os scores obtidos.
    """
    print(f"\n--- Avaliação com Cross-Validation ({cv}-fold) ---")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print("Scores individuais:", scores)
    print("Média:", scores.mean())
    print("Desvio:", scores.std())
    return scores



# ===================================================================
# 3. MODELOS OTIMIZADOS COM GRID SEARCH
# ===================================================================

def treinar_svm_otimizado(X_train, y_train):
    """
    Ajuste de hiperparâmetros do SVM usando GridSearchCV.
    """
    print("\n=== Ajustando SVM com GridSearchCV ===")

    param_grid = {
        "C": [0.1, 1, 5, 10, 50],
        "gamma": ["scale", 0.1, 0.01, 0.001],
        "kernel": ["rbf"]
    }

    grid = GridSearchCV(
        estimator=SVC(probability=True, random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Melhores parâmetros:", grid.best_params_)
    print("Melhor score médio CV:", grid.best_score_)

    return grid.best_estimator_


def treinar_mlp_otimizado(X_train, y_train):
    """
    Ajuste de hiperparâmetros do MLP (rede neural).
    """
    print("\n=== Ajustando MLP com GridSearchCV ===")

    param_grid = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "learning_rate_init": [0.001, 0.01],
        "alpha": [0.0001, 0.001],
        "activation": ["relu"]
    }

    grid = GridSearchCV(
        estimator=MLPClassifier(max_iter=600, random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Melhores parâmetros:", grid.best_params_)
    print("Melhor score CV:", grid.best_score_)

    return grid.best_estimator_



# ===================================================================
# 4. Geração do arquivo de submissão
# ===================================================================

def gerar_submissao(model, X_test, test_ids, label_encoder, base_path="dataset",
                    nome_arquivo="submission.csv"):
    """
    Gera submissão no formato exigido pela competição:

        Id,Predicted_0,Predicted_1,Predicted_2
    """
    print("\n--- Gerando arquivo de submissão ---")

    n_classes = len(label_encoder.classes_)

    # caso o modelo tenha predict_proba
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
    else:
        # fallback: usar predição dura
        preds = model.predict(X_test)
        y_pred_proba = np.zeros((len(preds), n_classes))
        for i, p in enumerate(preds):
            y_pred_proba[i, p] = 1.0

    prob_cols = [f"Predicted_{i}" for i in range(n_classes)]

    submission = pd.DataFrame(
        np.column_stack([test_ids, y_pred_proba]),
        columns=["Id"] + prob_cols
    )

    # normalização linha a linha
    submission[prob_cols] = submission[prob_cols].div(
        submission[prob_cols].sum(axis=1), axis=0
    )

    output_path = os.path.join(base_path, nome_arquivo)
    submission.to_csv(output_path, index=False, float_format="%.6f")

    print(f"\n✅ Submissão salva em: {output_path}")
    print("\nPrévia:")
    print(submission.head())

    return submission
