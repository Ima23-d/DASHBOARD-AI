# app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

DATA_PATH = "clientes_churn.csv"
MODEL_PATH = "modelo_churn.joblib"
FORECAST_PATH = "previsao_prox_mes.json"

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ============================================================
# LOADERS
# ============================================================
@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["mes"] = df["mes"].astype(str)
    return df


def patch_ohe(pipe):
    """
    Corrige incompatibilidade comum do OneHotEncoder entre versões do scikit-learn:
    - sparse_output (novo) vs sparse (antigo)
    """
    try:
        pre = pipe.named_steps.get("pre")
        cat = getattr(pre, "named_transformers_", {}).get("cat") if pre is not None else None
        ohe = getattr(cat, "named_steps", {}).get("onehot") if cat is not None else None
        if ohe is None:
            return pipe

        if hasattr(ohe, "sparse_output") and not hasattr(ohe, "sparse"):
            ohe.sparse = ohe.sparse_output
        if hasattr(ohe, "sparse") and not hasattr(ohe, "sparse_output"):
            ohe.sparse_output = ohe.sparse
    except Exception:
        pass
    return pipe


@st.cache_resource
def load_model(path: str = MODEL_PATH):
    pipe = joblib.load(path)
    return patch_ohe(pipe)


# ============================================================
# FEATURES
# ============================================================
def make_time_features(mes_str: str):
    per = pd.Period(mes_str, freq="M")
    m, y = per.month, per.year
    return y, m, np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)


def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    years, months, sins, coss = zip(*df_in["mes"].map(make_time_features))

    cols = [
        "idade",
        "regiao",
        "plano",
        "canal_aquisicao",
        "tenure_meses",
        "mensalidade",
        "uso_horas",
        "tickets_suporte",
        "atraso_pagamento_dias",
        "satisfacao",
    ]
    X = df_in[cols].copy()
    X["ano"] = years
    X["mes_num"] = months
    X["mes_sin"] = sins
    X["mes_cos"] = coss
    return X


# ============================================================
# SIDEBAR (FILTROS)
# ============================================================
df = load_data()

st.sidebar.title("Filtros")
meses = sorted(df["mes"].unique(), key=lambda x: pd.Period(x, freq="M"))
mes_sel = st.sidebar.selectbox("Mês", meses, index=len(meses) - 1)

planos = ["(todos)"] + sorted(df["plano"].unique().tolist())
plano_sel = st.sidebar.selectbox("Plano", planos, index=0)

regioes = ["(todas)"] + sorted(df["regiao"].unique().tolist())
regiao_sel = st.sidebar.selectbox("Região", regioes, index=0)

carregar_modelo = st.sidebar.toggle("Carregar modelo treinado", value=True)

df_f = df[df["mes"] == mes_sel].copy()
if plano_sel != "(todos)":
    df_f = df_f[df_f["plano"] == plano_sel]
if regiao_sel != "(todas)":
    df_f = df_f[df_f["regiao"] == regiao_sel]


# ============================================================
# MODEL LOAD
# ============================================================
model, model_error = None, None
if carregar_modelo:
    try:
        model = load_model()
    except Exception as e:
        model_error = e


# ============================================================
# HEADER / KPIs
# ============================================================
st.title("Churn Dashboard (Cancelamento de Clientes)")
st.caption("Filtros: mês, plano e região • Gráficos interativos (Plotly) • Previsão e explicação do churn")

clientes = int(df_f["customer_id"].nunique()) if len(df_f) else 0
churn_count = int(df_f["cancelou_prox_mes"].sum()) if len(df_f) else 0
churn_rate = float(df_f["cancelou_prox_mes"].mean()) if len(df_f) else 0.0
sat_avg = float(df_f["satisfacao"].mean()) if len(df_f) else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes (no mês)", f"{clientes}")
c2.metric("Cancelamentos (próx. mês)", f"{churn_count}")
c3.metric("Taxa de churn", f"{100 * churn_rate:.1f}%")
c4.metric("Satisfação média", "-" if np.isnan(sat_avg) else f"{sat_avg:.2f}")

st.divider()


# ============================================================
# FORECAST NEXT MONTH
# ============================================================
st.subheader("Previsão de grande cancelamento (próximo mês)")
try:
    with open(FORECAST_PATH, "r", encoding="utf-8") as f:
        fc = json.load(f)

    st.info(
        f"Base: **{fc['ultimo_mes_base']}** → Próximo mês: **{fc['proximo_mes']}** | "
        f"Previsão taxa: **{100 * fc['previsao_taxa_cancelamento']:.1f}%** | "
        f"Limiar (P75): **{100 * fc['limiar_alto_churn_p75']:.1f}%** | "
        f"Classe: **{fc['classe']}**"
    )
except Exception:
    st.warning("Arquivo de previsão não encontrado. Rode: `python train_model.py`")

st.divider()


# ============================================================
# AGGREGATIONS
# ============================================================
agg = (
    df.groupby("mes")
    .agg(
        clientes=("customer_id", "nunique"),
        churns=("cancelou_prox_mes", "sum"),
        churn_rate=("cancelou_prox_mes", "mean"),
        media_satisfacao=("satisfacao", "mean"),
        media_uso=("uso_horas", "mean"),
    )
    .reset_index()
)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(
    ["Visão Mensal", "Análise do Mês (Filtros)", "Explicação & Risco por Cliente"]
)

# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    st.subheader("Visão geral por mês (Jan–Dez)")

    colA, colB = st.columns(2)
    with colA:
        fig = px.line(agg, x="mes", y="churn_rate", markers=True, title="Taxa de churn por mês")
        fig.update_layout(xaxis_title="", yaxis_title="churn_rate", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig = px.bar(agg, x="mes", y="churns", title="Cancelamentos por mês")
        fig.update_layout(xaxis_title="", yaxis_title="cancelamentos", height=420)
        st.plotly_chart(fig, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        fig = px.line(agg, x="mes", y="media_satisfacao", markers=True, title="Satisfação média por mês")
        fig.update_layout(xaxis_title="", yaxis_title="satisfação", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with colD:
        fig = px.line(agg, x="mes", y="media_uso", markers=True, title="Uso médio (horas) por mês")
        fig.update_layout(xaxis_title="", yaxis_title="uso_horas", height=380)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 2
# -----------------------------
with tab2:
    st.subheader("Análise do mês selecionado (com filtros)")

    col1, col2 = st.columns(2)
    with col1:
        pie = (
            df_f[df_f["cancelou_prox_mes"] == 1]
            .groupby("regiao")["customer_id"]
            .nunique()
            .reset_index(name="churns")
        )
        if len(pie) == 0:
            st.info("Sem churn dentro dos filtros para montar o gráfico por região.")
        else:
            fig = px.pie(
                pie,
                names="regiao",
                values="churns",
                hole=0.55,
                title="Participação do churn por região (mês filtrado)",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df_f, x="mensalidade", nbins=20, title="Distribuição de mensalidade (mês filtrado)")
        fig.update_layout(height=420, xaxis_title="mensalidade", yaxis_title="frequência")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if len(df_f) == 0:
            st.info("Sem dados com os filtros atuais.")
        else:
            df_tmp = df_f.copy()
            df_tmp["status"] = np.where(df_tmp["cancelou_prox_mes"] == 1, "Cancelou", "Não cancelou")
            fig = px.box(df_tmp, x="status", y="satisfacao", title="Satisfação (Cancelou vs Não cancelou) – mês filtrado")
            fig.update_layout(height=380, xaxis_title="", yaxis_title="satisfação")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        grp = (
            df_f.groupby("plano")["cancelou_prox_mes"]
            .mean()
            .reset_index(name="churn_rate")
            .sort_values("churn_rate", ascending=False)
        )
        fig = px.bar(
            grp,
            x="plano",
            y="churn_rate",
            title="Taxa de churn por plano (mês filtrado)",
            text=grp["churn_rate"].map(lambda v: f"{100 * v:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, xaxis_title="", yaxis_title="churn_rate")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Relações úteis (mês filtrado)")
    col5, col6 = st.columns(2)

    with col5:
        if len(df_f) == 0:
            st.info("Sem dados com os filtros atuais.")
        else:
            df_tmp = df_f.copy()
            df_tmp["status"] = np.where(df_tmp["cancelou_prox_mes"] == 1, "Cancelou", "Não cancelou")
            fig = px.scatter(
                df_tmp,
                x="uso_horas",
                y="satisfacao",
                color="status",
                title="Uso vs Satisfação (colorido por churn)",
                hover_data=["customer_id", "plano", "regiao", "tickets_suporte", "atraso_pagamento_dias"],
            )
            fig.update_layout(height=420, xaxis_title="uso_horas", yaxis_title="satisfação")
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        if len(df_f) == 0:
            st.info("Sem dados com os filtros atuais.")
        else:
            fig = px.histogram(
                df_f,
                x="tickets_suporte",
                color=df_f["cancelou_prox_mes"].map({0: "Não cancelou", 1: "Cancelou"}),
                barmode="overlay",
                title="Distribuição de tickets (Cancelou vs Não cancelou)",
            )
            fig.update_layout(height=420, xaxis_title="tickets_suporte", yaxis_title="frequência")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 3
# -----------------------------
with tab3:
    st.subheader("Por que está cancelando? (explicação global)")
    st.caption("Coeficiente **positivo** ↑ aumenta chance de churn • **negativo** ↓ reduz chance")

    if model_error is not None:
        st.error(
            "Seu modelo foi carregado, mas deu erro ao usar `predict_proba`. "
            "Isso geralmente é versão diferente do scikit-learn entre treino e execução.\n\n"
            f"Erro: {model_error}\n\n"
            "✅ Solução recomendada: rode `python train_model.py` no MESMO ambiente do Streamlit."
        )

    if model is None:
        st.warning("Modelo não carregado. Rode: `python train_model.py` e tente de novo.")
    else:
        try:
            pre = model.named_steps["pre"]
            clf = model.named_steps["clf"]
            ohe = pre.named_transformers_["cat"].named_steps["onehot"]

            cat_cols = ["regiao", "plano", "canal_aquisicao"]
            num_cols = [
                "idade",
                "tenure_meses",
                "mensalidade",
                "uso_horas",
                "tickets_suporte",
                "atraso_pagamento_dias",
                "satisfacao",
                "ano",
                "mes_num",
                "mes_sin",
                "mes_cos",
            ]

            feat_names = num_cols + ohe.get_feature_names_out(cat_cols).tolist()
            coefs = clf.coef_.ravel()

            imp = pd.DataFrame({"feature": feat_names, "coef": coefs})
            top = (
                imp.assign(impacto_abs=lambda d: d["coef"].abs())
                .sort_values("impacto_abs", ascending=False)
                .head(15)
                .sort_values("coef")
            )

            fig = px.bar(
                top,
                x="coef",
                y="feature",
                orientation="h",
                title="Top 15 fatores (coeficientes) – direção do impacto no churn",
            )
            fig.update_layout(height=520, xaxis_title="coef", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver tabela (top fatores)"):
                st.dataframe(top[["feature", "coef"]], use_container_width=True)

        except Exception as e:
            st.warning(f"Não consegui extrair coeficientes/nomes das features do modelo. Erro: {e}")

        st.divider()

        st.subheader("Pontuação de risco por cliente (mês filtrado)")
        st.caption("Selecione um cliente e veja a probabilidade estimada de churn no próximo mês.")

        if len(df_f) == 0:
            st.info("Sem dados para o mês/filtros selecionados.")
        else:
            cid = st.selectbox("Cliente", sorted(df_f["customer_id"].unique().tolist()))
            row = df_f[df_f["customer_id"] == cid].head(1).copy()

            try:
                p = float(model.predict_proba(build_features(row))[0, 1])
                st.metric("Probabilidade de churn (próx. mês)", f"{100 * p:.1f}%")

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Plano", str(row["plano"].iloc[0]))
                k2.metric("Satisfação", f"{float(row['satisfacao'].iloc[0]):.2f}")
                k3.metric("Uso (h)", f"{float(row['uso_horas'].iloc[0]):.1f}")
                k4.metric("Tickets", f"{int(row['tickets_suporte'].iloc[0])}")

            except Exception as e:
                st.error(
                    "Erro ao calcular probabilidade. Isso confirma incompatibilidade do modelo/encoder.\n\n"
                    f"Erro: {e}\n\n"
                    "✅ Rode `python train_model.py` no mesmo ambiente e reinicie o Streamlit."
                )
