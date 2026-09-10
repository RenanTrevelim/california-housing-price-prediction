# California Housing Market Radar — Machine Learning e Inteligência Geográfica

Projeto de Ciência de Dados para análise e previsão de valores imobiliários na Califórnia utilizando **Machine Learning, Feature Engineering, otimização de hiperparâmetros e Explainable AI**.

Além da construção do modelo preditivo, o projeto transforma as estimativas em uma solução de **inteligência territorial**, permitindo identificar regiões compatíveis com diferentes critérios de orçamento, localização e posicionamento de mercado por meio de uma aplicação Streamlit.

---

## Visão Geral

O projeto busca responder à seguinte pergunta:

> É possível estimar o valor dos imóveis de uma região utilizando características econômicas, habitacionais e geográficas e transformar essas previsões em uma ferramenta útil para análise do mercado imobiliário?

A solução foi desenvolvida de ponta a ponta:

```text
Preparação dos Dados
        ↓
Análise Exploratória
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Otimização
        ↓
Validação e Generalização
        ↓
Explainable AI
        ↓
Inteligência Geográfica
        ↓
Aplicação Streamlit
```

O objetivo não é apenas obter boas métricas, mas construir uma solução capaz de conectar **modelo, localização e contexto de negócio**.

---

## Dataset

O projeto utiliza o dataset **California Housing**, composto por informações agregadas de diferentes regiões da Califórnia.

A base possui:

- **20.640 observações**
- **10 variáveis originais**
- **1 variável categórica**
- **207 valores ausentes** em `total_quartos`
- variável alvo: `valor_media_imovel`

Principais variáveis:

```text
longitude
latitude
idade_mediana_imoveis
total_comodos
total_quartos
populacao
domicilios
renda_media
proximidade_oceano
valor_media_imovel
```

As observações representam características regionais e não imóveis individuais.

---

## Estrutura do Projeto

```text
california-housing-price-prediction/
│
├── data/
│   ├── housing.csv
│   └── dados_housing.csv
│
├── models/
│   └── pipeline_completo.pkl
│
├── notebooks/
│   ├── 01_preparacao_dataset.ipynb
│   ├── 02_analise_exploratoria.ipynb
│   └── 03_modelos_machine_learning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── predict.py
│   └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 1. Preparação dos Dados

O primeiro notebook concentra as etapas necessárias para garantir consistência antes da análise e modelagem.

Foram realizadas:

- leitura e inspeção da base;
- tradução e padronização das variáveis;
- validação dos tipos de dados;
- análise de valores ausentes;
- verificação de duplicidades;
- investigação dos registros com `total_quartos` ausente;
- padronização da variável `proximidade_oceano`;
- exportação da base preparada.

Foram encontrados:

```text
207 valores ausentes em total_quartos
≈ 1% da base
```

Esses valores não foram preenchidos antecipadamente.

A imputação foi posteriormente incorporada ao pipeline de Machine Learning e ajustada somente sobre dados de treinamento, reduzindo o risco de **data leakage**.

---

# 2. Análise Exploratória

A análise exploratória buscou entender quais características estão associadas à valorização imobiliária.

Entre os principais resultados estão:

- forte relação entre renda e preço;
- diferenças relevantes entre interior e regiões costeiras;
- padrões espaciais evidentes em latitude e longitude;
- assimetria na distribuição do target;
- forte correlação entre algumas variáveis estruturais;
- concentração de observações próxima ao limite de **US$ 500 mil**.

## Distribuição do Valor dos Imóveis

A variável alvo apresenta distribuição assimétrica, com maior concentração nas faixas intermediárias e uma quantidade menor de regiões de alto valor.

Também existe uma concentração próxima de **US$ 500 mil**, sugerindo um limite superior presente no dataset.

<!-- GRÁFICO 1 — Histograma / Boxplot da variável alvo -->
<img width="582" height="454" alt="image" src="https://github.com/user-attachments/assets/04522110-0d18-4309-8cc9-a0ee6c06e9c9" />


---

## Localização como Fator de Valorização

Latitude e longitude revelaram um dos padrões mais importantes do projeto.

Ao posicionar as observações geograficamente e representar o valor dos imóveis pela cor, torna-se evidente que a valorização não ocorre de maneira uniforme pelo território.

Regiões próximas ao litoral apresentam maior concentração de valores elevados, enquanto grande parte das regiões do interior possui valores menores.

<!-- GRÁFICO 2 — Scatter Longitude x Latitude colorido pelo preço -->
<img width="844" height="539" alt="image" src="https://github.com/user-attachments/assets/5b13428e-9751-4303-8711-5cdafad5eb83" />


---

## Proximidade do Oceano

A variável `proximidade_oceano` reforçou a influência da localização.

As categorias analisadas foram:

```text
interior
menos_1h_oceano
proximo_oceano
proximo_baia
ilha
```

Regiões do `interior` apresentaram os menores valores médios e medianos, enquanto áreas próximas à baía e ao oceano apresentaram valores significativamente superiores.

A categoria `ilha` possui apenas cinco observações e, portanto, deve ser interpretada com cautela.

<!-- GRÁFICO 3 — Boxplot por proximidade do oceano -->
<img width="1161" height="571" alt="image" src="https://github.com/user-attachments/assets/724c5961-f6c0-4ca2-8504-cc4524d3e3f5" />


---

## Renda Média

A `renda_media` apresentou a associação numérica mais forte com o preço.

A correlação de Spearman encontrada foi aproximadamente:

```text
0.68
```

Regiões com maior renda tendem a apresentar imóveis mais valorizados.

Entretanto, a dispersão observada mostra que renda não explica o preço isoladamente, reforçando a necessidade de combinar informações econômicas, habitacionais e geográficas.

---

# 3. Feature Engineering

A análise exploratória motivou a criação de duas novas variáveis:

```text
comodos_por_domicilio
quartos_por_comodo
```

Definidas por:

```python
dados["comodos_por_domicilio"] = dados["total_comodos"] / dados["domicilios"]
dados["quartos_por_comodo"] = dados["total_quartos"] / dados["total_comodos"]
```

As novas features buscam representar características relativas das regiões, em vez de depender apenas de quantidades absolutas.

A utilidade dessas variáveis foi avaliada empiricamente.

| Cenário | R² | MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| XGBoost sem FE | 0.824 | 31.457 | 48.423 | 17.9% |
| XGBoost com FE | **0.829** | **31.022** | **47.707** | **17.7%** |

O ganho foi incremental, porém consistente para o melhor modelo inicial. Por esse motivo, o cenário com Feature Engineering foi mantido.

---

# 4. Estratégia de Modelagem

A base foi dividida em:

```text
60% → Treino
20% → Validação
20% → Teste
```

Como o target é contínuo, foram criadas temporariamente cinco faixas de preço utilizando `pd.qcut`.

Essas faixas foram utilizadas exclusivamente para preservar uma distribuição semelhante do target entre os conjuntos.

Após a divisão, a variável auxiliar foi removida antes do treinamento.

O conjunto de teste permaneceu isolado durante a seleção e otimização dos modelos.

---

## Pré-processamento

O pré-processamento foi construído com `Pipeline` e `ColumnTransformer`.

Para variáveis numéricas:

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler
```

Para `proximidade_oceano`:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder
```

Essa arquitetura permite reproduzir as mesmas transformações durante treinamento, validação e inferência.

---

# 5. Modelos Avaliados

Foram testados algoritmos com diferentes níveis de complexidade:

```text
Regressão Linear
Árvore de Decisão
Random Forest
KNN
XGBoost
```

Também foram avaliadas estratégias ensemble:

```text
Voting Regressor
Stacking Regressor
```

O **XGBoost apresentou o melhor desempenho inicial**, tornando-se o principal candidato para as etapas de otimização.

---

# 6. Otimização de Hiperparâmetros

Inicialmente foram realizados experimentos com:

```text
RandomizedSearchCV
```

nos modelos:

```text
XGBoost
Random Forest
KNN
```

Em seguida, o XGBoost foi submetido a uma busca mais direcionada utilizando **Optuna**.

Foram otimizados parâmetros relacionados a:

```text
n_estimators
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
gamma
reg_alpha
reg_lambda
```

O espaço de busca também foi ajustado para aumentar a regularização e reduzir sinais de overfitting.

---

## Optuna — 100 Trials

O estudo final utilizou:

```text
100 trials
5-fold Cross-Validation
MAE como função objetivo
```

No conjunto de validação, a configuração encontrada apresentou:

```text
R²   = 0.819
MAE  = US$ 30.639
RMSE = US$ 49.028
MAPE = 16.6%
```

A ampliação da busca melhorou principalmente o MAE e o MAPE.

Entre os hiperparâmetros com maior influência dentro do estudo estavam:

```text
n_estimators
learning_rate
min_child_weight
reg_lambda
```

<!-- GRÁFICO 4 — Optimization History / Hyperparameter Importance -->
<img width="1179" height="446" alt="image" src="https://github.com/user-attachments/assets/02292bd2-d471-4f52-a54f-d61251c11a92" />
<img width="1178" height="448" alt="image" src="https://github.com/user-attachments/assets/72c6135a-864d-4555-a870-cf23f6fc6db4" />



---

# 7. Comparação dos Modelos

Os principais candidatos apresentaram:

| Modelo | R² | MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| XGBoost Optuna — 100 | 0.819 | **30.639** | 49.028 | **16.6%** |
| XGBoost | **0.829** | 31.022 | **47.707** | 17.7% |
| Stacking | 0.822 | 31.635 | 48.726 | 17.7% |
| Voting | 0.818 | 31.981 | 49.174 | 18.2% |

O XGBoost original apresentou R² e RMSE ligeiramente melhores, enquanto o modelo otimizado apresentou menor MAE e MAPE.

Considerando as métricas prioritárias, a maior regularização e as análises posteriores de generalização, foi selecionado:

```text
XGBoost + Optuna — 100 Trials
```

<!-- GRÁFICO 5 — Comparação dos modelos por MAE -->
<img width="986" height="486" alt="image" src="https://github.com/user-attachments/assets/b0b508e4-20ca-4eac-9c6f-e5a8ac611cae" />


---

# 8. Validação e Generalização

A validação final foi realizada sobre o **pipeline completo**, garantindo que o pré-processamento fosse ajustado novamente dentro de cada fold.

```text
Dados de Treino
      ↓
KFold
      ↓
Pré-processamento
      ↓
XGBoost
      ↓
Validação do Fold
```

Resultados médios:

| Métrica | Média | Desvio Padrão |
|---|---:|---:|
| R² | 0.827 | 0.009 |
| MAE | 31.256 | 573 |
| RMSE | 47.998 | 811 |
| MAPE | 16.62% | 0.20% |

Os baixos desvios padrão indicam comportamento consistente entre os folds.

---

## Learning Curve

A Learning Curve foi utilizada para analisar a diferença entre desempenho de treino e validação.

No maior conjunto utilizado:

```text
MAE Treino     ≈ US$ 24.331
MAE Validação  ≈ US$ 31.258
Gap            ≈ US$ 6.927
```

A regularização reduziu significativamente o gap observado em experimentos anteriores.

O modelo apresentou um nível de overfitting **baixo a moderado e controlado**.

<!-- GRÁFICO 6 — Learning Curve -->
<img width="1161" height="569" alt="image" src="https://github.com/user-attachments/assets/c43a6b73-4723-4f72-a105-66b69e97d7e9" />


---

# 9. Avaliação Final no Teste

Após concluir todas as decisões de modelagem, o conjunto de teste foi utilizado pela primeira vez.

Resultados:

```text
R²   = 0.824
MAE  = US$ 30.596
RMSE = US$ 48.129
MAPE = 15.9%
```

Comparando validação cruzada e teste:

```text
Cross-Validation
R²   ≈ 0.827
MAE  ≈ US$ 31.256

Teste Final
R²   ≈ 0.824
MAE  ≈ US$ 30.596
```

A proximidade entre os resultados reforça a capacidade de generalização do modelo.

---

## Análise dos Resíduos

Os resíduos permanecem concentrados próximos de zero, embora exista maior dispersão em determinadas faixas de preço.

Os maiores erros aparecem principalmente entre regiões de maior valor.

Parte desse comportamento também está associada ao limite superior existente na variável alvo.

<!-- GRÁFICO 7 — Resíduos x Predições + Histograma -->
<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/a6f969be-fdd4-4d42-95ba-b2d6d8bcd980" />


<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/5308d351-3782-4bb9-8bd7-cdf016b9d05f" />




---

# 10. Interpretabilidade com SHAP

O desempenho do modelo foi complementado com técnicas de Explainable AI.

Foram utilizadas:

```text
Feature Importance
+
SHAP
```

A Feature Importance destacou principalmente características relacionadas à:

```text
proximidade do oceano
renda média
latitude
longitude
```

Já o SHAP permitiu analisar não apenas importância, mas também **direção e magnitude do impacto das features**.

---

## Importância Global

Entre as variáveis com maior impacto médio segundo SHAP estão:

```text
renda_media
longitude
latitude
proximidade_oceano_interior
populacao
```

A análise reforça os padrões encontrados durante a EDA: **renda e localização são componentes fundamentais das previsões**.

<!-- GRÁFICO 8 — SHAP Bar Plot -->
<img width="837" height="540" alt="image" src="https://github.com/user-attachments/assets/50edd0dd-192f-4eb5-a910-d1704dfb375d" />



---

## Direção dos Impactos

O SHAP Summary Plot mostrou que:

- valores elevados de `renda_media` tendem a elevar as previsões;
- pertencer ao `interior` tende a reduzir o valor previsto;
- latitude e longitude possuem efeitos não lineares;
- diferentes características interagem na formação das estimativas.

<!-- GRÁFICO 9 — SHAP Beeswarm -->
<img width="780" height="540" alt="image" src="https://github.com/user-attachments/assets/478f6c24-14b0-4069-a325-760f239ccaf4" />



---

## Explicação Individual

Também foi analisada uma previsão específica.

```text
Valor Real     ≈ US$ 176,6 mil
Valor Previsto ≈ US$ 142,0 mil
```

Entre os maiores impactos negativos estavam:

```text
proximidade_oceano_interior ≈ -US$ 33,0 mil
latitude                    ≈ -US$ 21,0 mil
populacao                    ≈ -US$ 17,7 mil
renda_media                  ≈ -US$ 8,6 mil
```

Enquanto a longitude contribuiu positivamente em aproximadamente:

```text
+US$ 28,2 mil
```

O exemplo demonstra como diferentes fatores econômicos, territoriais e habitacionais se combinam até formar uma previsão.

<!-- GRÁFICO 10 — SHAP Waterfall -->

<img width="1132" height="752" alt="image" src="https://github.com/user-attachments/assets/502b67f5-d04a-4ae0-beab-5005a1f4005f" />

---

# 11. Geração de Valor para o Negócio

A etapa seguinte foi transformar as previsões em uma solução que pudesse responder a um problema mais próximo do mercado.

Foi desenvolvido o:

# California Housing Market Radar

Uma ferramenta de **screening territorial** que permite explorar regiões da Califórnia de acordo com uma estratégia imobiliária.

O usuário pode definir:

```text
Orçamento máximo
Perfil geográfico
Segmento de mercado
```

A solução utiliza o modelo para identificar regiões aderentes e calcular:

```text
Valor estimado
Margem disponível
Uso do orçamento
Segmento
Perfil geográfico
Latitude
Longitude
```

Isso permite responder perguntas como:

> Quais regiões apresentam valores compatíveis com determinado orçamento?

> Onde essas regiões estão localizadas?

> Qual perfil geográfico predomina?

> Quais regiões estão mais próximas do limite financeiro definido?

---

## Inteligência Territorial

Latitude e longitude são utilizadas para transformar as previsões em uma visão territorial do mercado.

As regiões aderentes são destacadas sobre a distribuição espacial da Califórnia e coloridas de acordo com seu valor estimado.

Essa visualização representa um dos principais elementos da solução, pois conecta:

```text
Preço
+
Localização
+
Estratégia
```

<!-- GRÁFICO 11 — Market Radar geográfico -->
<img width="990" height="590" alt="image" src="https://github.com/user-attachments/assets/c73148ae-8386-4853-979b-0759abc43627" />


---

## Ranking de Aderência

As regiões selecionadas são ordenadas de acordo com a proximidade entre o valor estimado e o orçamento definido.

A interface apresenta informações como:

```text
Perfil geográfico
Segmento
Longitude
Latitude
Renda média
Valor estimado
Uso do orçamento
Margem disponível
```

O ranking funciona como ferramenta de **screening analítico**.

Ele não representa recomendação automática de investimento ou expectativa de retorno financeiro.

---

# 12. Pipeline Final

O pré-processamento e o modelo selecionado foram consolidados em um único pipeline:

```text
Feature Engineering
        ↓
ColumnTransformer
        ↓
Imputação
Padronização
One-Hot Encoding
        ↓
XGBoost
        ↓
Previsão
```

O modelo foi persistido utilizando `joblib`:

```python
joblib.dump(
    pipeline_final,
    "../models/pipeline_completo.pkl"
)
```

Artefato gerado:

```text
models/pipeline_completo.pkl
```

Essa estrutura permite reproduzir de maneira consistente o mesmo processamento utilizado durante o treinamento.

---

# 13. Aplicação Streamlit

A solução final foi disponibilizada por meio de uma aplicação Streamlit.

A camada de aplicação está organizada em:

```text
src/
├── __init__.py
├── predict.py
└── app.py
```

## `predict.py`

Responsável por:

- carregamento do pipeline;
- validação das entradas;
- criação das features;
- execução das previsões;
- preparação dos resultados.

## `app.py`

Responsável pela interface e pela camada de negócio.

A aplicação apresenta:

- visão geral do projeto;
- configuração automática da estratégia;
- indicadores executivos;
- inteligência territorial;
- leitura comercial;
- ranking de regiões;
- exportação dos resultados.

<!-- IMAGEM 12 — Tela principal da aplicação -->
<img width="1425" height="689" alt="image" src="https://github.com/user-attachments/assets/5321cca0-0d6e-4bf8-ad21-ac94d68aceba" />


---

# Limitações

Os resultados devem ser interpretados considerando algumas limitações:

- os registros representam regiões agregadas, e não imóveis individuais;
- a variável alvo possui um limite superior próximo de US$ 500 mil;
- o dataset representa padrões históricos;
- a aplicação não utiliza anúncios imobiliários em tempo real;
- Feature Importance e SHAP explicam o comportamento do modelo, mas não estabelecem causalidade.

Portanto, o Market Radar deve ser interpretado como uma ferramenta de **inteligência territorial e apoio à análise**, e não como recomendação financeira automática.

---

# Tecnologias

O projeto utiliza:

```text
Python
Pandas
NumPy
Scikit-learn
XGBoost
Optuna
SHAP
Matplotlib
Seaborn
Streamlit
Joblib
Jupyter Notebook
VS Code
Git
GitHub
```

---

# Como Executar

Clone o repositório:

```bash
git clone https://github.com/RenanTrevelim/california-housing-price-prediction.git
```

Acesse o projeto:

```bash
cd california-housing-price-prediction
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative no Windows:

```bash
.venv\Scripts\activate
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

---

## Notebooks

Execute na seguinte ordem:

```text
01_preparacao_dataset.ipynb
        ↓
02_analise_exploratoria.ipynb
        ↓
03_modelos_machine_learning.ipynb
```

---

## Streamlit

A partir da raiz do projeto:

```bash
python -m streamlit run src/app.py
```

A aplicação utiliza:

```text
data/dados_housing.csv
models/pipeline_completo.pkl
```

---

# Arquitetura da Solução

```text
California Housing
        ↓
Preparação e Validação
        ↓
EDA + Análise Geográfica
        ↓
Feature Engineering
        ↓
Pipeline de Pré-processamento
        ↓
Modelos de Machine Learning
        ↓
RandomizedSearchCV
        ↓
Optuna
        ↓
Cross-Validation
        ↓
Learning Curve
        ↓
Teste Final
        ↓
Explainable AI — SHAP
        ↓
Predições
        ↓
Inteligência Territorial
        ↓
Market Radar
        ↓
Streamlit
```

---

# Conclusão

O projeto percorre o ciclo completo de uma solução de Machine Learning aplicada ao mercado imobiliário, desde a preparação dos dados até a construção de uma aplicação orientada ao negócio.

O modelo final foi o:

```text
XGBoost otimizado com Optuna — 100 Trials
```

Resultados no conjunto de teste:

```text
R²   = 0.824
MAE  = US$ 30.596
RMSE = US$ 48.129
MAPE = 15.9%
```

A proximidade entre os resultados de Cross-Validation e teste, juntamente com a redução do gap observada na Learning Curve, reforçou a estabilidade e capacidade de generalização do modelo.

As análises de SHAP também mostraram que **renda, localização e proximidade do oceano** possuem papel importante nas previsões.

O principal diferencial do projeto, entretanto, está na transformação dessas previsões em uma ferramenta de inteligência geográfica.

O **California Housing Market Radar** conecta:

```text
Machine Learning
+
Explainable AI
+
Geospatial Analytics
+
Business Intelligence
+
Produto de Dados
```

demonstrando como um modelo preditivo pode evoluir de um experimento em notebook para uma solução interativa voltada à exploração e priorização territorial.

---

# Próximas Evoluções

Como próximos passos, o projeto pode ser expandido com:

- integração com dados imobiliários atualizados;
- anúncios reais de imóveis;
- mapas interativos;
- enriquecimento geográfico com infraestrutura e serviços;
- dados macroeconômicos;
- MLflow para rastreamento de experimentos;
- API com FastAPI;
- testes automatizados;
- Docker;
- monitoramento de drift;
- deploy em ambiente Cloud;
- pipeline automatizado de atualização e retreinamento.

---

# Autor

**Renan Assis Trevelim**

Projeto desenvolvido como aplicação prática de:

**Data Science • Machine Learning • Explainable AI • Geospatial Analytics • Business Intelligence • Streamlit**
