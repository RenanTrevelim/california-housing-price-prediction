from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.predict import prever_valores


# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================
st.set_page_config(
    page_title="California Housing Market Radar",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CAMINHOS
# ==================================================
ROOT = Path(__file__).resolve().parents[1]
CAMINHO_DADOS = ROOT / "data" / "dados_housing.csv"


# ==================================================
# CONFIGURAÇÕES
# ==================================================
ORDEM_SEGMENTOS = [
    "Econômico",
    "Intermediário",
    "Alto Padrão",
    "Premium",
]

NOMES_PROXIMIDADE = {
    "interior": "Interior",
    "ilha": "Ilha",
    "menos_1h_oceano": "Menos de 1h do oceano",
    "proximo_baia": "Próximo à baía",
    "proximo_oceano": "Próximo ao oceano",
}


# ==================================================
# HTML
# ==================================================
def renderizar_html(conteudo: str) -> None:
    st.html(dedent(conteudo).strip())


# ==================================================
# ESTILO VISUAL
# ==================================================
renderizar_html(
    """
    <style>

        .stApp,
        [data-testid="stAppViewContainer"] {
            background-color: #F4F7FC;
            color: #0F172A;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #0F172A 0%,
                #172554 100%
            );
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: #FFFFFF !important;
        }

        /* Hero */
        .hero {
            padding: 2.5rem 2.7rem;
            border-radius: 24px;

            background: linear-gradient(
                135deg,
                #1E3A8A 0%,
                #2563EB 55%,
                #06B6D4 100%
            );

            margin-bottom: 1.7rem;

            box-shadow:
                0 18px 45px
                rgba(30, 64, 175, 0.18);
        }

        .hero h1 {
            margin: 0;
            color: #FFFFFF !important;
            font-size: 2.7rem;
            line-height: 1.15;
        }

        .hero p {
            max-width: 980px;
            margin-top: 1rem;
            margin-bottom: 0;

            color: #E0F2FE !important;

            font-size: 1.05rem;
            line-height: 1.7;
        }

        .badge {
            display: inline-block;
            padding: 0.4rem 0.9rem;
            margin-bottom: 1rem;
            border-radius: 999px;

            background-color:
                rgba(255, 255, 255, 0.18);

            color: #FFFFFF !important;

            font-size: 0.8rem;
            font-weight: 800;
        }

        /* Títulos */
        .section-title {
            margin-top: 1.9rem;
            margin-bottom: 0.45rem;

            color: #0F172A !important;

            font-size: 1.45rem;
            font-weight: 800;
        }

        .section-subtitle {
            margin-bottom: 1.2rem;

            color: #64748B !important;

            font-size: 0.9rem;
            line-height: 1.6;
        }

        /* Cards gerais */
        .info-card {
            min-height: 190px;
            padding: 1.55rem;

            border: 1px solid #DCE4F0;
            border-radius: 18px;

            background: #FFFFFF;

            box-shadow:
                0 8px 25px
                rgba(15, 23, 42, 0.05);
        }

        .info-card h3 {
            margin-top: 0;
            color: #1E3A8A !important;
        }

        .info-card p {
            color: #475569 !important;
            line-height: 1.65;
        }

        /* Filtros */
        .filter-header {
            margin-bottom: 0.5rem;

            color: #1E3A8A !important;

            font-size: 0.82rem;
            font-weight: 800;

            text-transform: uppercase;
            letter-spacing: 0.04rem;
        }

        .filter-help {
            color: #64748B !important;
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .automatic-badge {
            display: inline-block;

            margin-bottom: 1rem;

            padding: 0.45rem 0.8rem;

            border-radius: 999px;

            background-color: #DCFCE7;

            color: #166534 !important;

            font-size: 0.8rem;
            font-weight: 700;
        }

        /* KPI */
        .kpi-card {
            min-height: 132px;
            padding: 1.25rem 1.35rem;

            border: 1px solid #E2E8F0;
            border-radius: 18px;

            background:
                linear-gradient(
                    145deg,
                    #FFFFFF,
                    #F8FAFC
                );

            box-shadow:
                0 8px 22px
                rgba(15, 23, 42, 0.05);
        }

        .kpi-label {
            color: #64748B !important;

            font-size: 0.76rem;
            font-weight: 800;

            letter-spacing: 0.04rem;
            text-transform: uppercase;
        }

        .kpi-value {
            margin-top: 0.45rem;

            color: #0F172A !important;

            font-size: 1.9rem;
            font-weight: 800;
        }

        .kpi-detail {
            margin-top: 0.35rem;

            color: #64748B !important;

            font-size: 0.8rem;
        }

        /* Leitura comercial */
        .commercial-card {
            min-height: 100%;
            padding: 1.6rem;

            border: 1px solid #BFDBFE;
            border-radius: 18px;

            background:
                linear-gradient(
                    145deg,
                    #EFF6FF 0%,
                    #FFFFFF 100%
                );

            box-shadow:
                0 8px 24px
                rgba(37, 99, 235, 0.07);
        }

        .commercial-card h3 {
            margin-top: 0;
            margin-bottom: 1.2rem;

            color: #1E3A8A !important;

            font-size: 1.15rem;
        }

        .commercial-card p {
            margin-bottom: 1.05rem;

            color: #475569 !important;

            font-size: 0.92rem;
            line-height: 1.7;
        }

        .commercial-highlight {
            color: #2563EB !important;
            font-weight: 800;
        }

        .commercial-label {
            display: block;

            margin-bottom: 0.15rem;

            color: #64748B !important;

            font-size: 0.72rem;
            font-weight: 800;

            text-transform: uppercase;
            letter-spacing: 0.04rem;
        }

        /* Top regiões */
        .ranking-card {
            min-height: 145px;

            padding: 1.3rem 1.4rem;

            border: 1px solid #E2E8F0;
            border-radius: 18px;

            background: #FFFFFF;

            box-shadow:
                0 7px 20px
                rgba(15, 23, 42, 0.05);
        }

        .ranking-position {
            margin-bottom: 0.5rem;

            color: #2563EB !important;

            font-size: 0.77rem;
            font-weight: 800;

            text-transform: uppercase;
        }

        .ranking-value {
            color: #0F172A !important;

            font-size: 1.5rem;
            font-weight: 800;
        }

        .ranking-detail {
            margin-top: 0.45rem;

            color: #64748B !important;

            font-size: 0.82rem;
            line-height: 1.55;
        }

        /* Tabela */
        [data-testid="stDataFrame"] {
            overflow: hidden;

            border: 1px solid #DCE4F0;
            border-radius: 16px;

            box-shadow:
                0 8px 22px
                rgba(15, 23, 42, 0.05);
        }

        /* Multiselect */
        [data-baseweb="tag"] {
            background-color: #2563EB !important;
        }

        /* Download */
        [data-testid="stDownloadButton"] button {
            width: 100%;

            border: none;
            border-radius: 11px;

            background:
                linear-gradient(
                    90deg,
                    #2563EB,
                    #06B6D4
                );

            color: #FFFFFF;

            font-weight: 700;
        }

        [data-testid="stDownloadButton"] button:hover {
            border: none;

            background:
                linear-gradient(
                    90deg,
                    #1D4ED8,
                    #0891B2
                );

            color: #FFFFFF;
        }

        /* Nota */
        .model-note {
            padding: 1.1rem 1.3rem;

            border-left: 5px solid #2563EB;
            border-radius: 14px;

            background-color: #EAF2FF;

            color: #1E3A8A !important;

            line-height: 1.65;
        }

        /* Rodapé */
        .footer {
            margin-top: 3rem;

            padding-top: 1.5rem;

            border-top:
                1px solid #DCE4F0;

            color: #64748B !important;

            font-size: 0.88rem;

            text-align: center;
        }

    </style>
    """
)


# ==================================================
# FUNÇÕES AUXILIARES
# ==================================================
def formatar_moeda(valor: float) -> str:
    if pd.isna(valor):
        return "-"

    return f"${valor:,.0f}"


def formatar_inteiro(valor: int) -> str:
    return f"{valor:,}"


def formatar_proximidade(valor: str) -> str:
    return NOMES_PROXIMIDADE.get(
        valor,
        valor.replace("_", " ").title(),
    )


def converter_csv(dados: pd.DataFrame) -> bytes:
    return dados.to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")


# ==================================================
# FUNÇÕES DE NEGÓCIO
# ==================================================
def criar_base_negocio(
    dados: pd.DataFrame,
) -> pd.DataFrame:

    resultado = dados.copy()

    resultado["valor_estimado"] = prever_valores(
        resultado
    )

    resultado["segmento_mercado"] = pd.qcut(
        resultado["valor_estimado"].rank(method="first"),
        q=4,
        labels=ORDEM_SEGMENTOS,
    )

    return resultado


def filtrar_regioes(
    dados: pd.DataFrame,
    orcamento: float,
    proximidades: list,
    segmentos: list,
) -> pd.DataFrame:

    resultado = dados[
        dados["proximidade_oceano"].isin(proximidades)
        & dados["segmento_mercado"].isin(segmentos)
        & (dados["valor_estimado"] <= orcamento)
    ].copy()

    resultado["margem_orcamento"] = (
        orcamento - resultado["valor_estimado"]
    )

    return resultado.sort_values(
        "valor_estimado",
        ascending=False,
    )


# ==================================================
# CARREGAMENTO DOS DADOS
# ==================================================
@st.cache_data(show_spinner=False)
def carregar_dados() -> pd.DataFrame:

    if not CAMINHO_DADOS.exists():
        raise FileNotFoundError(
            f"Base não encontrada: {CAMINHO_DADOS}"
        )

    return pd.read_csv(
        CAMINHO_DADOS
    )


@st.cache_data(show_spinner=False)
def carregar_base_negocio() -> pd.DataFrame:

    dados = carregar_dados()

    return criar_base_negocio(
        dados
    )


# ==================================================
# GRÁFICO GEOGRÁFICO
# ==================================================
def criar_grafico_geografico(
    base: pd.DataFrame,
    regioes: pd.DataFrame,
    orcamento: float,
):

    demais_regioes = base.loc[
        ~base.index.isin(regioes.index)
    ]

    fig, ax = plt.subplots(
        figsize=(16, 8.5)
    )

    fig.patch.set_facecolor(
        "#FFFFFF"
    )

    ax.set_facecolor(
        "#F8FAFC"
    )

    sns.scatterplot(
        data=demais_regioes,
        x="longitude",
        y="latitude",
        color="#CBD5E1",
        alpha=0.24,
        s=20,
        linewidth=0,
        label="Demais regiões",
        ax=ax,
    )

    sns.scatterplot(
        data=regioes,
        x="longitude",
        y="latitude",
        hue="valor_estimado",
        palette="viridis",
        alpha=0.90,
        s=40,
        edgecolor="#FFFFFF",
        linewidth=0.25,
        legend="brief",
        ax=ax,
    )

    ax.set_title(
        (
            "Radar Geográfico de Regiões "
            f"até {formatar_moeda(orcamento)}"
        ),
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
        pad=20,
    )

    ax.set_xlabel(
        "Longitude",
        fontsize=10,
        color="#475569",
    )

    ax.set_ylabel(
        "Latitude",
        fontsize=10,
        color="#475569",
    )

    ax.tick_params(
        colors="#64748B",
        labelsize=9,
    )

    ax.grid(
        alpha=0.10
    )

    ax.spines[
        [
            "top",
            "right",
            "left",
            "bottom",
        ]
    ].set_visible(False)

    ax.legend(
        title="Valor estimado",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        fontsize=8,
        title_fontsize=9,
        loc="upper right",
    )

    plt.tight_layout()

    return fig


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:

    st.markdown(
        "## 🏠 Market Radar"
    )

    st.caption(
        "California Housing Intelligence"
    )

    st.divider()

    pagina = st.radio(
        "Navegação",
        [
            "Visão geral",
            "Radar de mercado",
            "Sobre o projeto",
        ],
    )

    st.divider()

    st.markdown(
        "### Tecnologia"
    )

    st.markdown(
        """
**Machine Learning**

XGBoost + Optuna

**Explicabilidade**

SHAP

**Produto**

Inteligência geográfica
        """
    )

    st.divider()

    st.caption(
        "Solução analítica para "
        "prospecção e análise regional."
    )


# ==================================================
# VISÃO GERAL
# ==================================================
if pagina == "Visão geral":

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Data Product • Real Estate Intelligence
            </span>

            <h1>
                California Housing Market Radar
            </h1>

            <p>
                Um produto analítico que utiliza Machine Learning
                e inteligência geográfica para identificar regiões
                da Califórnia alinhadas ao orçamento, localização
                e posicionamento de mercado definidos pelo usuário.
            </p>

        </div>
        """
    )

    renderizar_html(
        """
        <div class="section-title">
            Do modelo à oportunidade comercial
        </div>

        <div class="section-subtitle">
            O modelo deixa de ser apenas uma previsão numérica
            e passa a apoiar decisões sobre onde concentrar
            esforços de análise e prospecção.
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        renderizar_html(
            """
            <div class="info-card">

                <h3>📍 Onde procurar</h3>

                <p>
                    Localiza regiões geográficas que atendem
                    aos critérios comerciais selecionados,
                    reduzindo o universo de análise.
                </p>

            </div>
            """
        )

    with col2:

        renderizar_html(
            """
            <div class="info-card">

                <h3>💰 Quanto esperar</h3>

                <p>
                    O modelo estima o valor mediano esperado
                    para cada região e permite compará-lo com
                    um teto de orçamento.
                </p>

            </div>
            """
        )

    with col3:

        renderizar_html(
            """
            <div class="info-card">

                <h3>🎯 Onde priorizar</h3>

                <p>
                    Segmentação, localização e valor estimado
                    são combinados para construir um radar inicial
                    de regiões para prospecção.
                </p>

            </div>
            """
        )

    renderizar_html(
        """
        <div class="section-title">
            Proposta de valor
        </div>
        """
    )

    renderizar_html(
        """
        <div class="model-note">

            Uma imobiliária, incorporadora ou equipe comercial
            pode possuir milhares de regiões possíveis para análise.

            O Market Radar reduz esse universo utilizando
            localização, características regionais e estimativas
            de preço para indicar onde concentrar uma investigação
            comercial mais aprofundada.

        </div>
        """
    )


# ==================================================
# RADAR DE MERCADO
# ==================================================
elif pagina == "Radar de mercado":

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Geographic Market Intelligence
            </span>

            <h1>
                Radar de Mercado Imobiliário
            </h1>

            <p>
                Defina o perfil desejado e identifique
                automaticamente regiões da Califórnia
                compatíveis com sua estratégia comercial.
                Todos os indicadores são recalculados
                automaticamente conforme os filtros mudam.
            </p>

        </div>
        """
    )

    try:

        with st.spinner(
            "Carregando modelo e analisando mercado..."
        ):

            base_negocio = carregar_base_negocio()

        proximidades_disponiveis = sorted(
            base_negocio["proximidade_oceano"]
            .dropna()
            .unique()
        )

        segmentos_disponiveis = list(
            base_negocio[
                "segmento_mercado"
            ].cat.categories
        )

        # ==================================================
        # FILTROS
        # ==================================================
        renderizar_html(
            """
            <div class="section-title">
                Defina o mercado-alvo
            </div>

            <div class="section-subtitle">
                Ajuste orçamento, localização e segmento.
                A análise é atualizada automaticamente.
            </div>

            <span class="automatic-badge">
                ● Atualização automática
            </span>
            """
        )

        filtro1, filtro2, filtro3 = st.columns(
            [1.15, 1, 1]
        )

        with filtro1:

            renderizar_html(
                """
                <div class="filter-header">
                    1 • Orçamento máximo
                </div>
                """
            )

            orcamento = st.slider(
                "Teto de investimento",
                min_value=50_000,
                max_value=500_000,
                value=250_000,
                step=10_000,
                format="$%d",
                label_visibility="collapsed",
            )

            renderizar_html(
                """
                <div class="filter-help">
                    Define o maior valor estimado
                    considerado na prospecção.
                </div>
                """
            )

        with filtro2:

            renderizar_html(
                """
                <div class="filter-header">
                    2 • Perfil geográfico
                </div>
                """
            )

            proximidades = st.multiselect(
                "Localização",
                options=proximidades_disponiveis,
                default=proximidades_disponiveis,
                format_func=formatar_proximidade,
                label_visibility="collapsed",
            )

            renderizar_html(
                """
                <div class="filter-help">
                    Selecione as regiões geográficas
                    de interesse comercial.
                </div>
                """
            )

        with filtro3:

            renderizar_html(
                """
                <div class="filter-header">
                    3 • Segmento de mercado
                </div>
                """
            )

            segmentos = st.multiselect(
                "Segmentos",
                options=segmentos_disponiveis,
                default=segmentos_disponiveis,
                label_visibility="collapsed",
            )

            renderizar_html(
                """
                <div class="filter-help">
                    Direcione a busca para o posicionamento
                    imobiliário desejado.
                </div>
                """
            )

        if not proximidades:

            st.warning(
                "Selecione pelo menos "
                "um perfil geográfico."
            )

            st.stop()

        if not segmentos:

            st.warning(
                "Selecione pelo menos "
                "um segmento de mercado."
            )

            st.stop()

        # ==================================================
        # FILTRAGEM
        # ==================================================
        base_filtrada = base_negocio[
            base_negocio[
                "proximidade_oceano"
            ].isin(proximidades)
            & base_negocio[
                "segmento_mercado"
            ].isin(segmentos)
        ].copy()

        regioes = filtrar_regioes(
            base_negocio,
            orcamento,
            proximidades,
            segmentos,
        )

        total_analisado = len(
            base_filtrada
        )

        total_regioes = len(
            regioes
        )

        cobertura = (
            total_regioes
            / total_analisado
            if total_analisado > 0
            else 0
        )

        valor_mediano = (
            regioes[
                "valor_estimado"
            ].median()
            if total_regioes > 0
            else 0
        )

        # ==================================================
        # KPIs
        # ==================================================
        renderizar_html(
            """
            <div class="section-title">
                Mercado encontrado
            </div>

            <div class="section-subtitle">
                Dimensão do mercado regional compatível
                com os critérios selecionados.
            </div>
            """
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Universo analisado
                    </div>

                    <div class="kpi-value">
                        {formatar_inteiro(total_analisado)}
                    </div>

                    <div class="kpi-detail">
                        regiões dentro do perfil selecionado
                    </div>

                </div>
                """
            )

        with kpi2:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Regiões encontradas
                    </div>

                    <div class="kpi-value">
                        {formatar_inteiro(total_regioes)}
                    </div>

                    <div class="kpi-detail">
                        dentro do orçamento
                    </div>

                </div>
                """
            )

        with kpi3:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Cobertura
                    </div>

                    <div class="kpi-value">
                        {cobertura:.1%}
                    </div>

                    <div class="kpi-detail">
                        do universo analisado
                    </div>

                </div>
                """
            )

        with kpi4:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Valor mediano
                    </div>

                    <div class="kpi-value">
                        {formatar_moeda(valor_mediano)}
                    </div>

                    <div class="kpi-detail">
                        das regiões selecionadas
                    </div>

                </div>
                """
            )

        # ==================================================
        # RESULTADOS
        # ==================================================
        if regioes.empty:

            st.warning(
                "Nenhuma região foi encontrada "
                "para os filtros selecionados."
            )

        else:

            proximidade_dominante = (
                regioes[
                    "proximidade_oceano"
                ]
                .mode()
                .iloc[0]
            )

            segmento_dominante = (
                regioes[
                    "segmento_mercado"
                ]
                .mode()
                .iloc[0]
            )

            menor_valor = (
                regioes[
                    "valor_estimado"
                ].min()
            )

            maior_valor = (
                regioes[
                    "valor_estimado"
                ].max()
            )

            renda_mediana = (
                regioes[
                    "renda_media"
                ].median()
            )

            margem_mediana = (
                regioes[
                    "margem_orcamento"
                ].median()
            )

            # ==================================================
            # MAPA + LEITURA COMERCIAL
            # ==================================================
            renderizar_html(
                """
                <div class="section-title">
                    Onde estão as oportunidades de prospecção?
                </div>

                <div class="section-subtitle">
                    O radar transforma latitude e longitude
                    em uma visão territorial das regiões
                    compatíveis com a estratégia definida.
                </div>
                """
            )

            mapa_col, leitura_col = st.columns(
                [2.55, 1]
            )

            with mapa_col:

                fig = criar_grafico_geografico(
                    base_filtrada,
                    regioes,
                    orcamento,
                )

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

            with leitura_col:

                renderizar_html(
                    f"""
                    <div class="commercial-card">

                        <h3>
                            💡 Leitura comercial
                        </h3>

                        <p>
                            <span class="commercial-label">
                                Mercado disponível
                            </span>

                            <span class="commercial-highlight">
                                {formatar_inteiro(total_regioes)}
                                regiões
                            </span>

                            atendem aos critérios definidos,
                            representando
                            <span class="commercial-highlight">
                                {cobertura:.1%}
                            </span>
                            do universo analisado.
                        </p>

                        <p>
                            <span class="commercial-label">
                                Perfil dominante
                            </span>

                            O segmento mais frequente é
                            <span class="commercial-highlight">
                                {segmento_dominante}
                            </span>

                            e a localização predominante é
                            <span class="commercial-highlight">
                                {formatar_proximidade(
                                    proximidade_dominante
                                )}
                            </span>.
                        </p>

                        <p>
                            <span class="commercial-label">
                                Faixa estimada
                            </span>

                            Os valores previstos variam entre
                            <span class="commercial-highlight">
                                {formatar_moeda(menor_valor)}
                            </span>

                            e

                            <span class="commercial-highlight">
                                {formatar_moeda(maior_valor)}
                            </span>.
                        </p>

                        <p>
                            <span class="commercial-label">
                                Perfil econômico regional
                            </span>

                            A renda mediana das regiões
                            selecionadas é
                            <span class="commercial-highlight">
                                {renda_mediana:.2f}
                            </span>.
                        </p>

                        <p>
                            <span class="commercial-label">
                                Espaço no orçamento
                            </span>

                            A margem mediana até o teto
                            definido é
                            <span class="commercial-highlight">
                                {formatar_moeda(margem_mediana)}
                            </span>.
                        </p>

                    </div>
                    """
                )

            # ==================================================
            # TOP 3
            # ==================================================
            renderizar_html(
                """
                <div class="section-title">
                    Regiões para iniciar a prospecção
                </div>

                <div class="section-subtitle">
                    Regiões cujo valor estimado está mais
                    próximo do teto definido, sem ultrapassá-lo.
                </div>
                """
            )

            top3 = (
                regioes
                .head(3)
                .reset_index(drop=True)
            )

            top_cols = st.columns(3)

            medalhas = [
                "🥇 Região 01",
                "🥈 Região 02",
                "🥉 Região 03",
            ]

            for indice, coluna in enumerate(
                top_cols
            ):

                if indice < len(top3):

                    linha = top3.iloc[
                        indice
                    ]

                    with coluna:

                        renderizar_html(
                            f"""
                            <div class="ranking-card">

                                <div class="ranking-position">
                                    {medalhas[indice]}
                                </div>

                                <div class="ranking-value">
                                    {formatar_moeda(
                                        linha["valor_estimado"]
                                    )}
                                </div>

                                <div class="ranking-detail">

                                    📍
                                    {formatar_proximidade(
                                        linha["proximidade_oceano"]
                                    )}

                                    <br>

                                    🏷️
                                    {linha["segmento_mercado"]}

                                    <br>

                                    🌐
                                    {linha["latitude"]:.2f},
                                    {linha["longitude"]:.2f}

                                    <br>

                                    💵 Margem:
                                    {formatar_moeda(
                                        linha["margem_orcamento"]
                                    )}

                                </div>

                            </div>
                            """
                        )

            # ==================================================
            # TABELA
            # ==================================================
            renderizar_html(
                """
                <div class="section-title">
                    Regiões candidatas para prospecção
                </div>

                <div class="section-subtitle">
                    Explore as coordenadas das regiões
                    selecionadas e aprofunde a análise
                    comercial individualmente.
                </div>
                """
            )

            tabela = (
                regioes
                .head(30)
                .copy()
                .reset_index(drop=True)
            )

            tabela.insert(
                0,
                "ranking",
                range(
                    1,
                    len(tabela) + 1,
                ),
            )

            tabela[
                "proximidade_oceano"
            ] = tabela[
                "proximidade_oceano"
            ].map(
                formatar_proximidade
            )

            tabela[
                "localizacao"
            ] = (
                "https://www.google.com/maps?q="
                + tabela[
                    "latitude"
                ].astype(str)
                + ","
                + tabela[
                    "longitude"
                ].astype(str)
            )

            colunas_exibicao = [
                "ranking",
                "proximidade_oceano",
                "segmento_mercado",
                "valor_estimado",
                "renda_media",
                "latitude",
                "longitude",
                "margem_orcamento",
                "localizacao",
            ]

            st.dataframe(
                tabela[
                    colunas_exibicao
                ],
                use_container_width=True,
                hide_index=True,
                height=570,
                column_config={

                    "ranking":
                    st.column_config.NumberColumn(
                        "#",
                        format="%d",
                        width="small",
                    ),

                    "proximidade_oceano":
                    st.column_config.TextColumn(
                        "Perfil geográfico",
                        width="medium",
                    ),

                    "segmento_mercado":
                    st.column_config.TextColumn(
                        "Segmento",
                        width="medium",
                    ),

                    "valor_estimado":
                    st.column_config.NumberColumn(
                        "Valor estimado",
                        format="$ %.0f",
                    ),

                    "renda_media":
                    st.column_config.NumberColumn(
                        "Renda média",
                        format="%.2f",
                    ),

                    "latitude":
                    st.column_config.NumberColumn(
                        "Latitude",
                        format="%.2f",
                    ),

                    "longitude":
                    st.column_config.NumberColumn(
                        "Longitude",
                        format="%.2f",
                    ),

                    "margem_orcamento":
                    st.column_config.NumberColumn(
                        "Margem",
                        format="$ %.0f",
                    ),

                    "localizacao":
                    st.column_config.LinkColumn(
                        "Explorar região",
                        display_text="📍 Abrir mapa",
                    ),
                },
            )

            st.caption(
                "O Market Radar realiza screening regional. "
                "As posições apresentadas não representam "
                "recomendação financeira ou avaliação "
                "individual de propriedades."
            )

            # ==================================================
            # EXPORTAÇÃO
            # ==================================================
            exportacao = (
                regioes.copy()
            )

            exportacao[
                "proximidade_oceano"
            ] = exportacao[
                "proximidade_oceano"
            ].map(
                formatar_proximidade
            )

            st.download_button(
                "⬇️ Exportar regiões para prospecção",
                data=converter_csv(
                    exportacao
                ),
                file_name=(
                    "market_radar_regioes.csv"
                ),
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

    except Exception as erro:

        st.error(
            "Não foi possível executar "
            f"a análise: {erro}"
        )


# ==================================================
# SOBRE O PROJETO
# ==================================================
else:

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                End-to-End Data Science Project
            </span>

            <h1>
                Sobre o Market Radar
            </h1>

            <p>
                Projeto de portfólio que demonstra como
                transformar um modelo de Machine Learning
                em um produto analítico orientado à
                identificação e priorização de mercados regionais.
            </p>

        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    🤖 Machine Learning
                </h3>

                <p>
                    Preparação dos dados, análise exploratória,
                    Feature Engineering, Scikit-learn,
                    XGBoost, Optuna, Cross-validation,
                    análise de resíduos e SHAP.
                </p>

            </div>
            """
        )

    with col2:

        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    🏢 Produto de dados
                </h3>

                <p>
                    As previsões são transformadas em
                    segmentação, filtros comerciais,
                    inteligência geográfica, screening regional
                    e apoio à prospecção.
                </p>

            </div>
            """
        )

    renderizar_html(
        """
        <div class="section-title">
            Arquitetura da solução
        </div>
        """
    )

    st.code(
        """
dados_housing.csv
        ↓
Feature Engineering
        ↓
Pipeline Scikit-learn
        ↓
XGBoost + Optuna
        ↓
Valor regional estimado
        ↓
Segmentação de mercado
        ↓
Filtros comerciais
        ↓
Latitude + Longitude
        ↓
Market Radar
        ↓
Regiões candidatas para prospecção
        """,
        language="text",
    )

    renderizar_html(
        """
        <div class="model-note">

            <strong>Escopo do produto:</strong>

            o California Housing contém informações
            agregadas de regiões censitárias.

            Portanto, o Market Radar identifica
            <strong>regiões candidatas para prospecção</strong>
            e não imóveis individuais disponíveis para compra.

        </div>
        """
    )

    renderizar_html(
        """
        <div class="section-title">
            Tecnologias
        </div>
        """
    )

    st.code(
        """Python
Pandas
NumPy
Scikit-learn
XGBoost
Optuna
SHAP
Matplotlib
Seaborn
Streamlit
Joblib""",
        language="text",
    )


# ==================================================
# RODAPÉ
# ==================================================
renderizar_html(
    """
    <div class="footer">

        California Housing Market Radar •
        Machine Learning + Geographic Intelligence

    </div>
    """
)