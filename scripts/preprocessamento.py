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

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. Geração do dataset consolidado (wearables + train + users_info)
# ---------------------------------------------------------
def gerar_dataset(base_path: str) -> pd.DataFrame:
    """Lê dados dos sensores em 'wearables/', calcula estatísticas e junta com train.csv e users_info.txt."""
    print("\n--- Gerando dataset consolidado ---")

    wearables_path = os.path.join(base_path, "wearables")
    train_path = os.path.join(base_path, "train.csv")
    users_info_path = os.path.join(base_path, "users_info.txt")

    # # Checagens básicas
    # if not os.path.exists(train_path):
    #     raise FileNotFoundError("❌ Arquivo train.csv não encontrado.")
    # if not os.path.exists(users_info_path):
    #     raise FileNotFoundError("❌ Arquivo users_info.txt não encontrado.")
    # if not os.path.exists(wearables_path):
    #     raise FileNotFoundError("❌ Pasta 'wearables/' não encontrada.")

    train = pd.read_csv(train_path)
    users_info = pd.read_csv(users_info_path, sep=",")

    data_list = []

    for idx, row in train.iterrows():
        user_id = row["Id"]
        label = row["Label"]
        user_folder = os.path.join(wearables_path, user_id)

        # if not os.path.exists(user_folder):
        #     print(f"⚠️ Pasta {user_id} não encontrada, pulando.")
        #     continue

        user_data = {"Id": user_id, "classe": label}

        # Loop pelos sensores
        for sensor_file in os.listdir(user_folder):
            if not sensor_file.endswith(".csv"):
                continue

            sensor_name = os.path.splitext(sensor_file)[0]
            sensor_path = os.path.join(user_folder, sensor_file)

            if os.path.getsize(sensor_path) == 0:
                continue  # ignora arquivos vazios

            df_sensor = pd.read_csv(sensor_path)
            numeric_cols = df_sensor.select_dtypes(include=np.number)
            if numeric_cols.empty:
                continue

            # Estatísticas por sensor (escolher relevantes??)
            user_data[f"{sensor_name}_mean"] = numeric_cols.mean().mean()
            # user_data[f"{sensor_name}_std"] = numeric_cols.std().mean()
            # user_data[f"{sensor_name}_max"] = numeric_cols.max().max()
            # user_data[f"{sensor_name}_min"] = numeric_cols.min().min()

        data_list.append(user_data)

    df_features = pd.DataFrame(data_list)
    df = df_features
    # df = pd.merge(df_features, users_info, on="Id", how="left")

    print(f"Dataset com {df.shape[0]} amostras e {df.shape[1]} atributos.")
    return df

# ---------------------------------------------------------
# Geração do dataset de TESTE
# ---------------------------------------------------------

def gerar_dataset_teste(base_path: str) -> pd.DataFrame:
    """Gera o dataset consolidado de TESTE a partir de test.csv + wearables + users_info.txt."""
    print("\n--- Gerando dataset de TESTE ---")

    wearables_path = os.path.join(base_path, "wearables")
    test_path = os.path.join(base_path, "test.csv")
    users_info_path = os.path.join(base_path, "users_info.txt")

    # # Checagens básicas
    # if not os.path.exists(test_path):
    #     raise FileNotFoundError("❌ Arquivo test.csv não encontrado.")
    # if not os.path.exists(users_info_path):
    #     raise FileNotFoundError("❌ Arquivo users_info.txt não encontrado.")
    # if not os.path.exists(wearables_path):
    #     raise FileNotFoundError("❌ Pasta 'wearables/' não encontrada.")

    # Leitura dos dados
    test = pd.read_csv(test_path)
    users_info = pd.read_csv(users_info_path, sep=",")

    data_list = []

    for idx, row in test.iterrows():
        user_id = row["Id"]
        user_folder = os.path.join(wearables_path, user_id)

        # if not os.path.exists(user_folder):
        #     print(f"⚠️ Pasta {user_id} não encontrada, pulando.")
        #     continue

        user_data = {"Id": user_id}

        # Loop pelos sensores
        for sensor_file in os.listdir(user_folder):
            if not sensor_file.endswith(".csv"):
                continue

            sensor_name = os.path.splitext(sensor_file)[0]
            sensor_path = os.path.join(user_folder, sensor_file)

            if os.path.getsize(sensor_path) == 0:
                continue  # ignora arquivos vazios

            df_sensor = pd.read_csv(sensor_path)
            numeric_cols = df_sensor.select_dtypes(include=np.number)
            if numeric_cols.empty:
                continue

            # Estatísticas básicas por sensor
            user_data[f"{sensor_name}_mean"] = numeric_cols.mean().mean()
            user_data[f"{sensor_name}_std"] = numeric_cols.std().mean()
            user_data[f"{sensor_name}_max"] = numeric_cols.max().max()
            user_data[f"{sensor_name}_min"] = numeric_cols.min().min()

        data_list.append(user_data)

    df_features = pd.DataFrame(data_list)

    # Junta com informações pessoais
    df = pd.merge(df_features, users_info, on="Id", how="left")

    print(f"Dataset de TESTE com {df.shape[0]} amostras e {df.shape[1]} atributos.")
    return df


# ---------------------------------------------------------
# 2. Tratamento de valores ausentes
# ---------------------------------------------------------
def tratar_valores_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Tratando valores ausentes ---")
    df.replace("?", np.nan, inplace=True)

    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            moda = df[col].mode()
            if not moda.empty:
                df[col].fillna(moda[0], inplace=True)
            else:
                df[col].fillna("Desconhecido", inplace=True)
    return df


# ---------------------------------------------------------
# 3. Normalização
# ---------------------------------------------------------
def normalizar_dados(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Normalizando atributos numéricos ---")
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        print("⚠️ Nenhum atributo numérico encontrado para normalização.")
        return df
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


# ---------------------------------------------------------
# 4. Remoção de outliers
# ---------------------------------------------------------
def remover_outliers(df: pd.DataFrame, zscore_limite=3, proporcao_limite=0.1) -> pd.DataFrame:
    """
    Remove amostras com proporção de outliers acima do limite definido.
    Ex: proporcao_limite=0.1 → remove se mais de 10% das colunas forem outliers.
    """
    print("\n--- Removendo outliers ---")
    from scipy import stats

    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        print("⚠️ Nenhuma coluna numérica para detecção de outliers.")
        return df

    z_scores = np.abs(stats.zscore(num_df))
    proporcao_outliers = (z_scores > zscore_limite).mean(axis=1)

    filtro = proporcao_outliers < proporcao_limite
    df_filtrado = df[filtro]

    print(f"Removidos {len(df) - len(df_filtrado)} amostras com outliers.")
    return df_filtrado


def remover_outliers_IQR(df: pd.DataFrame, limite=1.5) -> pd.DataFrame:
    """
    Remove outliers com base no IQR (Interquartile Range).
    """
    print("\n--- Removendo outliers via IQR ---")
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        print("⚠️ Nenhuma coluna numérica para detecção de outliers.")
        return df

    Q1 = num_df.quantile(0.25)
    Q3 = num_df.quantile(0.75)
    IQR = Q3 - Q1

    filtro = ~((num_df < (Q1 - limite * IQR)) | (num_df > (Q3 + limite * IQR))).any(axis=1)
    df_filtrado = df[filtro]

    print(f"Removidos {len(df) - len(df_filtrado)} outliers pelo método IQR.")
    return df_filtrado

