# train_model.py
# Treina 2 modelos:
# 1) Classificação por cliente: probabilidade de "cancelou_prox_mes"
# 2) Previsão agregada: taxa de cancelamento do próximo mês (e classificação "alto/normal")

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "clientes_churn.csv"
OUT_MODEL = "modelo_churn.joblib"
OUT_FORECAST = "previsao_prox_mes.json"

def make_time_features(mes_str: str):
    # mes_str: "YYYY-MM"
    per = pd.Period(mes_str, freq="M")
    m = per.month
    y = per.year
    # sazonalidade simples
    sin = np.sin(2*np.pi*m/12)
    cos = np.cos(2*np.pi*m/12)
    return y, m, sin, cos

def main():
    df = pd.read_csv(DATA_PATH)

    # -------------------------
    # 1) MODELO DE CHURN (cliente)
    # -------------------------
    target = "cancelou_prox_mes"
    features = [
        "idade","regiao","plano","canal_aquisicao",
        "tenure_meses","mensalidade","uso_horas",
        "tickets_suporte","atraso_pagamento_dias","satisfacao",
        "mes"
    ]

    # features de tempo (ano/mês + sin/cos)
    y = df[target].astype(int)
    years, months, sins, coss = zip(*df["mes"].map(make_time_features))
    df_feat = df[features].copy()
    df_feat["ano"] = years
    df_feat["mes_num"] = months
    df_feat["mes_sin"] = sins
    df_feat["mes_cos"] = coss
    df_feat.drop(columns=["mes"], inplace=True)

    num_cols = ["idade","tenure_meses","mensalidade","uso_horas","tickets_suporte","atraso_pagamento_dias","satisfacao","ano","mes_num","mes_sin","mes_cos"]
    cat_cols = ["regiao","plano","canal_aquisicao"]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", clf)
    ])

    # Split temporal simples: últimos 3 meses como teste
    df["_ord"] = pd.PeriodIndex(df["mes"], freq="M").astype("int64")

    cutoff = np.sort(df["_ord"].unique())[-3]
    train_idx = df["_ord"] < cutoff
    test_idx = df["_ord"] >= cutoff

    X_train = df_feat.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = df_feat.loc[test_idx]
    y_test = y.loc[test_idx]

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)

    print("\n=== Modelo churn (cliente) ===")
    print(f"AUC: {auc:.3f} | Accuracy: {acc:.3f}")
    print("Matriz de confusão:\n", confusion_matrix(y_test, pred))
    print("\nRelatório:\n", classification_report(y_test, pred, digits=3))

    joblib.dump(pipe, OUT_MODEL)

    # -------------------------
    # 2) MODELO DE PREVISÃO: TAXA DE CHURN DO PRÓXIMO MÊS
    # -------------------------
    # Agrega por mês e prevê churn_rate do próximo mês (regressor), depois classifica "alto" vs "normal".
    agg = df.groupby("mes").agg(
        clientes=("customer_id","nunique"),
        churns=("cancelou_prox_mes","sum"),
        churn_rate=("cancelou_prox_mes","mean"),
        media_satisfacao=("satisfacao","mean"),
        media_uso=("uso_horas","mean"),
        media_atraso=("atraso_pagamento_dias","mean"),
        media_tickets=("tickets_suporte","mean"),
    ).reset_index()

    # cria alvo: churn_rate do mês seguinte
    agg["churn_rate_next"] = agg["churn_rate"].shift(-1)
    agg = agg.dropna().copy()

    y2 = agg["churn_rate_next"].values
    yyr, mm, sins, coss = zip(*agg["mes"].map(make_time_features))
    X2 = pd.DataFrame({
        "ano": yyr,
        "mes_num": mm,
        "mes_sin": sins,
        "mes_cos": coss,
        "clientes": agg["clientes"].values,
        "media_satisfacao": agg["media_satisfacao"].values,
        "media_uso": agg["media_uso"].values,
        "media_atraso": agg["media_atraso"].values,
        "media_tickets": agg["media_tickets"].values,
    })

    reg = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=2
    )
    reg.fit(X2, y2)

    # previsão para o "próximo mês" após o último disponível em df
    last_mes = pd.Period(df["mes"].max(), freq="M")
    next_mes = (last_mes + 1).strftime("%Y-%m")

    # features do último mês para projetar o próximo
    last_row = df[df["mes"] == str(last_mes)].copy()
    last_agg = {
        "clientes": int(last_row["customer_id"].nunique()),
        "media_satisfacao": float(last_row["satisfacao"].mean()),
        "media_uso": float(last_row["uso_horas"].mean()),
        "media_atraso": float(last_row["atraso_pagamento_dias"].mean()),
        "media_tickets": float(last_row["tickets_suporte"].mean()),
    }

    yyr, mm, sin, cos = make_time_features(next_mes)
    X_next = pd.DataFrame([{
        "ano": yyr, "mes_num": mm, "mes_sin": sin, "mes_cos": cos,
        **last_agg
    }])

    pred_rate = float(reg.predict(X_next)[0])
    # define "alto churn" como acima do percentil 75 histórico do churn_rate
    threshold = float(np.quantile(df.groupby("mes")["cancelou_prox_mes"].mean().values, 0.75))
    classe = "ALTO" if pred_rate >= threshold else "NORMAL"

    out = {
        "ultimo_mes_base": str(last_mes),
        "proximo_mes": next_mes,
        "previsao_taxa_cancelamento": pred_rate,
        "limiar_alto_churn_p75": threshold,
        "classe": classe
    }

    with open(OUT_FORECAST, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== Previsão (agregado) ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()