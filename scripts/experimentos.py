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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# ===================================================================
# 1. Treinamento dos modelos
# ===================================================================

def treinar_knn(X_train, y_train, k=7):
    print(f"\n--- Treinando KNN (k={k}) ---")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def treinar_naive_bayes(X_train, y_train):
    print("\n--- Treinando Naive Bayes (Gaussian) ---")
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def treinar_regressao_logistica(X_train, y_train):
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
# 2. Geração do arquivo de submissão
# ===================================================================

def gerar_submissao(model, X_test, test_ids, label_encoder, base_path="dataset",
                    nome_arquivo="submission.csv"):
    """
    Gera submissão no formato exigido pela competição.

    Formato:
        Id,Predicted_0,Predicted_1,Predicted_2
    """
    print("\n--- Gerando arquivo de submissão ---")

    n_classes = len(label_encoder.classes_)

    # =====================================================
    # Probabilidades obtidas do modelo
    # =====================================================
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)

    else:
        # Para modelos sem predict_proba
        preds = model.predict(X_test)
        y_pred_proba = np.zeros((len(preds), n_classes))
        for i, p in enumerate(preds):
            y_pred_proba[i, p] = 1.0

    # =====================================================
    # DataFrame de submissão
    # =====================================================
    prob_cols = [f"Predicted_{i}" for i in range(n_classes)]

    submission = pd.DataFrame(
        np.column_stack([test_ids, y_pred_proba]),
        columns=["Id"] + prob_cols
    )

    # Normalização linha a linha (soma = 1)
    submission[prob_cols] = submission[prob_cols].div(
        submission[prob_cols].sum(axis=1), axis=0
    )

    # =====================================================
    # Salvar arquivo
    # =====================================================
    output_path = os.path.join(base_path, nome_arquivo)
    submission.to_csv(output_path, index=False, float_format="%.6f")

    print(f"\n✅ Submissão salva em: {output_path}")
    print("\nPrévia da submissão:")
    print(submission.head())

    return submission