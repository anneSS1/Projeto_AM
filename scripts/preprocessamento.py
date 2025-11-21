# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
# Nome: Anne Mari Suenaga Sakai
# RA: 822304
# 
# Nome: Felipe Jun Nishitani
# RA: 822353
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats


# ================================================================
# 0. PREPARAÇÃO FINAL DOS DADOS (Encode, One-Hot, Normalização)
# ================================================================
def preparar_dados(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Prepara os dados de treino e teste para treinamento de modelos:
        ✔ Encode da variável alvo
        ✔ One-hot encoding dos atributos categóricos (fit no treino)
        ✔ Normalização numérica (fit no treino)
        ✔ Alinhamento das colunas entre treino e teste

    Retorna:
        X_train, y_train, X_test, test_ids, label_encoder, scaler
    """

    print("\n--- Preparando dados de treino e teste ---")

    # -------------------------------
    # 1) Encode da classe (treino)
    # -------------------------------
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train["classe"])

    # -------------------------------
    # 2) Seleção de atributos
    # -------------------------------
    X_train = df_train.drop(columns=["Id", "classe"], errors="ignore").copy()
    X_test = df_test.drop(columns=["Id"], errors="ignore").copy()
    test_ids = df_test["Id"].copy()

    # -------------------------------
    # 3) One-hot encoding
    # -------------------------------
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)

    # Alinha colunas (garante compatibilidade entre treino e teste)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    # -------------------------------
    # 4) Normalização
    # -------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    print(f"Treino shape final: {X_train_scaled.shape}")
    print(f"Teste shape final : {X_test_scaled.shape}")

    return X_train_scaled, y_train, X_test_scaled, test_ids, label_encoder, scaler



# ================================================================
# 1. GERAÇÃO DO DATASET DE TREINO (wearables + train.csv)
# ================================================================
def gerar_dataset(base_path: str) -> pd.DataFrame:
    """
    Lê a pasta 'wearables/' e extrai estatísticas dos sensores para cada usuário.
    Junta com os rótulos presentes em train.csv.

    ⚠️ Usa as mesmas features que o dataset de TESTE para evitar inconsistências.
    """

    print("\n--- Gerando dataset consolidado (TREINO) ---")

    wearables_path = os.path.join(base_path, "wearables")
    train_path = os.path.join(base_path, "train.csv")
    users_info_path = os.path.join(base_path, "users_info.txt")

    train = pd.read_csv(train_path)
    users_info = pd.read_csv(users_info_path, sep=",")

    data_list = []

    for _, row in train.iterrows():
        user_id = row["Id"]
        label = row["Label"]
        user_folder = os.path.join(wearables_path, user_id)

        user_data = {"Id": user_id, "classe": label}

        # Cada arquivo .csv dentro da pasta corresponde a um sensor
        for sensor_file in os.listdir(user_folder):
            if not sensor_file.endswith(".csv"):
                continue

            sensor_name = os.path.splitext(sensor_file)[0]
            sensor_path = os.path.join(user_folder, sensor_file)

            if os.path.getsize(sensor_path) == 0:
                continue

            df_sensor = pd.read_csv(sensor_path)
            numeric_cols = df_sensor.select_dtypes(include=np.number)

            if numeric_cols.empty:
                continue

            # Usar as mesmas estatísticas para treino e teste
            user_data[f"{sensor_name}_mean"] = numeric_cols.mean().mean()
            user_data[f"{sensor_name}_std"] = numeric_cols.std().mean()
            user_data[f"{sensor_name}_max"] = numeric_cols.max().max()
            user_data[f"{sensor_name}_min"] = numeric_cols.min().min()

        data_list.append(user_data)

    df_features = pd.DataFrame(data_list)

    # Opcional: juntar com informações do usuário
    # df_features = pd.merge(df_features, users_info, on="Id", how="left")

    print(f"Treino consolidado: {df_features.shape[0]} amostras, {df_features.shape[1]} atributos.")
    return df_features



# ================================================================
# 2. GERAÇÃO DO DATASET DE TESTE (wearables + test.csv)
# ================================================================
def gerar_dataset_teste(base_path: str) -> pd.DataFrame:
    """Gera o dataset consolidado de TESTE, igual ao treino (mesmas features)."""

    print("\n--- Gerando dataset de TESTE ---")

    wearables_path = os.path.join(base_path, "wearables")
    test_path = os.path.join(base_path, "test.csv")
    users_info_path = os.path.join(base_path, "users_info.txt")

    test = pd.read_csv(test_path)
    users_info = pd.read_csv(users_info_path, sep=",")

    data_list = []

    for _, row in test.iterrows():
        user_id = row["Id"]
        user_folder = os.path.join(wearables_path, user_id)

        user_data = {"Id": user_id}

        for sensor_file in os.listdir(user_folder):
            if not sensor_file.endswith(".csv"):
                continue

            sensor_name = os.path.splitext(sensor_file)[0]
            sensor_path = os.path.join(user_folder, sensor_file)

            if os.path.getsize(sensor_path) == 0:
                continue

            df_sensor = pd.read_csv(sensor_path)
            numeric_cols = df_sensor.select_dtypes(include=np.number)

            if numeric_cols.empty:
                continue

            # Mesmas features do treino!
            user_data[f"{sensor_name}_mean"] = numeric_cols.mean().mean()
            user_data[f"{sensor_name}_std"] = numeric_cols.std().mean()
            user_data[f"{sensor_name}_max"] = numeric_cols.max().max()
            user_data[f"{sensor_name}_min"] = numeric_cols.min().min()

        data_list.append(user_data)

    df_features = pd.DataFrame(data_list)

    # df_features = pd.merge(df_features, users_info, on="Id", how="left")

    print(f"Teste consolidado: {df_features.shape[0]} amostras, {df_features.shape[1]} atributos.")
    return df_features



# ================================================================
# 3. TRATAMENTO DE VALORES AUSENTES
# ================================================================
def tratar_valores_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Tratando valores ausentes ---")

    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            moda = df[col].mode()
            df[col].fillna(moda[0] if not moda.empty else "Desconhecido", inplace=True)

    return df



# ================================================================
# 4. REMOÇÃO DE OUTLIERS (opcional)
# ================================================================
def remover_outliers(df: pd.DataFrame, z_lim=3, prop_lim=0.1) -> pd.DataFrame:
    """
    Remove linhas cujo percentual de outliers (Z-score) ultrapassa o limite.
    """
    print("\n--- Removendo outliers (Z-score) ---")

    num = df.select_dtypes(include=np.number)
    if num.empty:
        print("Nenhuma coluna numérica detectada.")
        return df

    z_scores = np.abs(stats.zscore(num))
    proporcao_outliers = (z_scores > z_lim).mean(axis=1)
    filtro = proporcao_outliers < prop_lim

    df_filtrado = df[filtro]
    print(f"Removidas {len(df) - len(df_filtrado)} amostras.")
    return df_filtrado


def remover_outliers_IQR(df: pd.DataFrame, limite=1.5) -> pd.DataFrame:
    """
    Remove outliers usando o método IQR.
    """
    print("\n--- Removendo outliers (IQR) ---")

    num = df.select_dtypes(include=np.number)
    if num.empty:
        print("Nenhuma coluna numérica detectada.")
        return df

    Q1 = num.quantile(0.25)
    Q3 = num.quantile(0.75)
    IQR = Q3 - Q1

    filtro = ~((num < (Q1 - limite * IQR)) | (num > (Q3 + limite * IQR))).any(axis=1)
    df_filtrado = df[filtro]

    print(f"Removidas {len(df) - len(df_filtrado)} amostras.")
    return df_filtrado


# ================================================================
# 5. REMOÇÃO DE ATRIBUTOS COM VARIÂNCIA ZERO
# ================================================================
def remover_variancia_zero(df):
    print("\n--- Removendo atributos com variância zero ---")
    variancia = df.var(numeric_only=True)
    cols_zero = variancia[variancia == 0].index.tolist()

    if cols_zero:
        print(f"Removidos {len(cols_zero)} atributos sem variância:", cols_zero)
        df = df.drop(columns=cols_zero)

    return df


# ================================================================
# 6. AJUSTE DE SKEW (assimetria)
# ================================================================
def ajustar_skew(df, limite=1.0):
    print("\n--- Corrigindo skew em atributos assimétricos ---")
    skew = df.select_dtypes(include=np.number).skew()

    cols_skew = skew[abs(skew) > limite].index

    for col in cols_skew:
        df[col] = np.log1p(df[col] - df[col].min() + 1)

    print("Transformados:", list(cols_skew))
    return df
