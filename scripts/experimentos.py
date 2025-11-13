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

# Arquivo com todas as funções e códigos referentes aos experimentos


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------------------------------------------------------------
# 1. Preparar dados de treino e teste
# -------------------------------------------------------------
def preparar_dados(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Recebe os DataFrames de treino e teste já pré-processados e prepara para os modelos."""
    print("\n--- Preparando dados de treino e teste ---")

    # Codificar classes alvo
    label_encoder = LabelEncoder()
    df_train["classe_cod"] = label_encoder.fit_transform(df_train["classe"])

    # Separa atributos e rótulos
    X_train = df_train.drop(columns=["Id", "classe", "classe_cod"], errors="ignore")
    y_train = df_train["classe_cod"]

    # Converte colunas categóricas em numéricas (one-hot encoding)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(df_test.drop(columns=["Id"], errors="ignore"), drop_first=True)

    # Alinha colunas entre treino e teste
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Normaliza dados numéricos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    return X_train, y_train, X_test, df_test["Id"], label_encoder


# -------------------------------------------------------------
# 2. Treinamento de modelos
# -------------------------------------------------------------
def treinar_knn(X_train, y_train, k=7):
    """Treina o modelo KNN e retorna o classificador."""
    print(f"\n--- Treinando modelo KNN (k={k}) ---")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def treinar_naive_bayes(X_train, y_train, tipo="gaussian"):
    """Treina o modelo Naive Bayes (Gaussian ou Categorical)."""
    print(f"\n--- Treinando modelo Naive Bayes ({tipo}) ---")
    model = GaussianNB(var_smoothing=1e-9)  
    model.fit(X_train, y_train)
    return model


def treinar_regressao_logistica(X_train, y_train):
    print("\n--- Treinando modelo de Regressão Logística ---")
    model = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def treinar_rede_neural(X_train, y_train):
    print("\n--- Treinando Rede Neural Artificial ---")
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model


def treinar_svm(X_train, y_train):
    print("\n--- Treinando Máquina de Vetores de Suporte (SVM) ---")
    model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
    model.fit(X_train, y_train)
    return model

# -------------------------------------------------------------
# 3. Avaliar modelo
# -------------------------------------------------------------
def avaliar_modelo(model, X, y, dataset_name="Treino"):
    """Exibe métricas de avaliação do modelo."""
    print(f"\n--- Avaliando modelo no conjunto de {dataset_name} ---")
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y, y_pred))
    return acc


# -------------------------------------------------------------
# 4. Gerar arquivo de submissão
# -------------------------------------------------------------
def gerar_submissao(model, X_test, test_ids, base_path="dataset", nome_arquivo="submission.csv"):
    """
    Aplica o modelo no conjunto de teste, salva o arquivo CSV no formato exigido e retorna o DataFrame.
    
    Formato exigido:
        Id,Predicted_0,Predicted_1,Predicted_2
        U_19341,0.33,0.33,0.33
        U_54670,0.99,0.01,0.00
        U_21920,0.24,0.56,0.20
    """
    print("\n--- Gerando arquivo de submissão ---")

    # Predição das probabilidades
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
    else:
        # Se o modelo não tiver predict_proba (ex: alguns classificadores), cria distribuição uniforme
        preds = model.predict(X_test)
        y_pred_proba = np.zeros((len(preds), 3))
        for i, p in enumerate(preds):
            y_pred_proba[i, p] = 1.0

    # Garante que test_ids seja Série ou DataFrame
    if isinstance(test_ids, pd.DataFrame):
        ids = test_ids["Id"]
    else:
        ids = pd.Series(test_ids, name="Id")

    # Monta o DataFrame no formato correto
    submission = pd.DataFrame({
        "Id": ids,
        "Predicted_0": y_pred_proba[:, 0],
        "Predicted_1": y_pred_proba[:, 1],
        "Predicted_2": y_pred_proba[:, 2],
    })

    # Normaliza (garante soma = 1)
    submission[["Predicted_0", "Predicted_1", "Predicted_2"]] = (
        submission[["Predicted_0", "Predicted_1", "Predicted_2"]].div(
            submission[["Predicted_0", "Predicted_1", "Predicted_2"]].sum(axis=1), axis=0
        )
    )

    # Salva CSV no formato solicitado
    output_path = os.path.join(base_path, nome_arquivo)
    submission.to_csv(output_path, index=False, float_format="%.6f")
    display(submission.head())

    return submission