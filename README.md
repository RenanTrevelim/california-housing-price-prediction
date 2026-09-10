# California Housing Market Radar — Machine Learning e Inteligência Geográfica

Projeto de Ciência de Dados para análise e previsão de valores imobiliários na Califórnia utilizando **Machine Learning, Feature Engineering, otimização de hiperparâmetros e interpretabilidade com SHAP**.

Além da modelagem preditiva, o projeto transforma as previsões em uma solução de **inteligência territorial**, permitindo explorar regiões compatíveis com diferentes critérios de orçamento, localização e posicionamento de mercado por meio de uma aplicação Streamlit.

---

## Objetivo

O projeto busca responder:

> É possível estimar o valor dos imóveis de uma região a partir de características econômicas, habitacionais e geográficas e transformar essas previsões em uma ferramenta útil para análise do mercado imobiliário?

O fluxo desenvolvido contempla:

```text
Preparação dos Dados
        ↓
Análise Exploratória
        ↓
Feature Engineering
        ↓
Modelagem e Otimização
        ↓
Validação
        ↓
Interpretabilidade
        ↓
Inteligência Geográfica
        ↓
Aplicação Streamlit
```

---

## Dataset

O projeto utiliza o dataset **California Housing**, composto por informações agregadas de regiões da Califórnia.

A base possui:

- **20.640 observações**
- **10 variáveis originais**
- **207 valores ausentes** em `total_quartos`
- variável alvo: `valor_media_imovel`

Principais atributos:

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

Durante a análise também foi identificada uma concentração de registros próxima de **US$ 500 mil**, indicando um limite superior presente na variável alvo.

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

O primeiro notebook concentra as etapas de validação e preparação da base.

Foram realizadas:

- tradução e padronização das variáveis;
- análise de tipos e estrutura;
- verificação de dados nulos e duplicados;
- investigação dos valores ausentes;
- padronização de `proximidade_oceano`;
- exportação da base preparada.

Os **207 valores ausentes em `total_quartos`** foram preservados nesta etapa e posteriormente tratados dentro do pipeline de Machine Learning, evitando data leakage.

---

# 2. Análise Exploratória

A análise exploratória mostrou que **localização e renda possuem forte relação com o valor dos imóveis**.

Entre os principais resultados:

- `renda_media` apresentou correlação de Spearman próxima de **0,68** com o target;
- regiões do `interior` apresentaram valores significativamente menores;
- regiões próximas ao oceano e à baía concentraram valores mais elevados;
- latitude e longitude revelaram padrões espaciais relevantes;
- variáveis relacionadas ao tamanho das regiões apresentaram forte correlação entre si.

A análise geográfica se tornou um dos principais elementos do projeto.

<!-- Adicione aqui uma imagem do scatterplot geográfico -->
<img width="1200" alt="Distribuição geográfica dos valores imobiliários" src="COLE_AQUI_O_LINK_DA_IMAGEM" />

---

# 3. Feature Engineering

Foram criadas duas novas variáveis:

```text
comodos_por_domicilio
quartos_por_comodo
```

As features representam relações proporcionais entre características habitacionais.

Sua utilidade foi avaliada comparando os mesmos modelos com e sem Feature Engineering.

O principal ganho ocorreu no XGBoost:

| Cenário | R² | MAE |
|---|---:|---:|
| Sem Feature Engineering | 0.824 | 31.457 |
| Com Feature Engineering | **0.829** | **31.022** |

Com base nesse resultado, as novas features foram mantidas nas etapas seguintes.

---

# 4. Estratégia de Modelagem

Os dados foram divididos em:

```text
60% Treino
20% Validação
20% Teste
```

Como o problema é de regressão, foram criadas temporariamente cinco faixas de preço com `pd.qcut` para preservar uma distribuição semelhante do target entre os conjuntos.

Essa variável auxiliar foi utilizada apenas na divisão dos dados e removida antes da modelagem.

O pré-processamento foi estruturado com:

```text
Variáveis Numéricas
SimpleImputer → StandardScaler

Variável Categórica
SimpleImputer → OneHotEncoder
```

Todo o fluxo foi organizado utilizando `Pipeline` e `ColumnTransformer`.

---

# 5. Modelos Avaliados

Foram comparados:

- Regressão Linear
- Árvore de Decisão
- Random Forest
- KNN
- XGBoost
- Voting Regressor
- Stacking Regressor

O XGBoost apresentou o melhor desempenho entre os modelos iniciais.

Também foram utilizadas estratégias de tuning com:

```text
RandomizedSearchCV
Optuna
```

---

# 6. Otimização com Optuna

O XGBoost foi submetido a uma busca mais ampla de hiperparâmetros utilizando **Optuna + KFold com 5 folds**.

Foram explorados parâmetros como:

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

A busca foi direcionada para reduzir overfitting e melhorar a capacidade de generalização.

O estudo final utilizou:

```text
100 trials
```

Resultado no conjunto de validação:

| Modelo | R² | MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| XGBoost Optuna — 100 | 0.819 | **30.639** | 49.028 | **16.6%** |
| XGBoost | **0.829** | 31.022 | **47.707** | 17.7% |
| Stacking | 0.822 | 31.635 | 48.726 | 17.7% |
| Voting | 0.818 | 31.981 | 49.174 | 18.2% |

O **XGBoost otimizado com 100 trials** foi selecionado pelo equilíbrio entre erro absoluto, regularização e generalização.

<!-- Adicione aqui a imagem da comparação dos modelos -->
<img width="1200" alt="Comparação dos modelos por MAE" src="COLE_AQUI_O_LINK_DA_IMAGEM" />

---

# 7. Validação e Generalização

A validação cruzada foi executada sobre o **pipeline completo**, garantindo que o pré-processamento fosse reajustado dentro de cada fold.

Resultados médios:

| Métrica | Média | Desvio Padrão |
|---|---:|---:|
| R² | 0.827 | 0.009 |
| MAE | 31.256 | 573 |
| RMSE | 47.998 | 811 |
| MAPE | 16.62% | 0.20% |

A baixa variabilidade entre os folds indica comportamento consistente do modelo.

## Learning Curve

No maior conjunto avaliado:

```text
MAE Treino     ≈ US$ 24.331
MAE Validação  ≈ US$ 31.258
Gap            ≈ US$ 6.927
```

O aumento da regularização reduziu significativamente a diferença entre treino e validação, indicando um nível de overfitting **baixo a moderado e controlado**.

<!-- Adicione aqui a imagem da Learning Curve -->
<img width="1200" alt="Learning Curve do pipeline final" src="COLE_AQUI_O_LINK_DA_IMAGEM" />

---

# 8. Avaliação Final

O conjunto de teste permaneceu isolado durante todo o processo de desenvolvimento.

Resultados finais:

```text
R²   = 0.824
MAE  = US$ 30.596
RMSE = US$ 48.129
MAPE = 15.9%
```

Os resultados ficaram próximos aos observados durante a Cross-Validation, reforçando a capacidade de generalização do modelo.

A análise dos resíduos também mostrou concentração próxima de zero, embora existam erros maiores nas faixas mais altas de preço e influência do limite superior presente no target.

---

# 9. Interpretabilidade com SHAP

A interpretabilidade do modelo foi analisada utilizando:

```text
Feature Importance
+
SHAP
```

As variáveis com maior impacto global incluíram:

```text
renda_media
longitude
latitude
proximidade_oceano_interior
populacao
```

Os resultados mostraram que:

- maior `renda_media` tende a elevar as previsões;
- regiões do `interior` tendem a reduzir o valor previsto;
- latitude e longitude possuem efeitos não lineares;
- fatores geográficos e econômicos atuam de forma combinada.

Também foi utilizada uma explicação individual com **SHAP Waterfall**, permitindo visualizar quanto cada variável contribuiu para uma previsão específica.

<!-- Adicione aqui a imagem do SHAP -->
<img width="1200" alt="Interpretabilidade do modelo com SHAP" src="COLE_AQUI_O_LINK_DA_IMAGEM" />

---

# 10. California Housing Market Radar

A principal evolução do projeto foi transformar o modelo em uma ferramenta de **inteligência territorial aplicada ao mercado imobiliário**.

A aplicação permite explorar regiões de acordo com:

```text
Orçamento máximo
Perfil geográfico
Segmento de mercado
```

A partir das previsões, são calculados indicadores como:

- valor estimado;
- margem disponível;
- segmento de mercado;
- perfil geográfico;
- utilização do orçamento.

O objetivo é responder perguntas como:

> Quais regiões estão dentro do orçamento disponível?

> Onde essas regiões estão localizadas?

> Quais regiões aproveitam melhor o orçamento sem ultrapassá-lo?

> Qual perfil geográfico predomina entre as oportunidades encontradas?

A ferramenta funciona como um **radar de mercado e screening territorial**, não como uma recomendação automática de investimento.

<!-- Adicione aqui a imagem do Market Radar -->
<img width="1400" alt="California Housing Market Radar" src="COLE_AQUI_O_LINK_DA_IMAGEM" />

---

# 11. Pipeline e Aplicação Streamlit

O modelo final foi consolidado com o pré-processamento e serializado utilizando `joblib`:

```text
models/pipeline_completo.pkl
```

A aplicação está organizada em:

```text
src/
├── predict.py
└── app.py
```

### `predict.py`

Responsável por:

- carregamento do pipeline;
- criação das features;
- validação dos dados;
- execução das previsões.

### `app.py`

Responsável pela interface Streamlit e pela camada de inteligência comercial.

A aplicação apresenta:

- visão geral do projeto;
- configuração automática da estratégia;
- indicadores executivos;
- análise geográfica;
- leitura comercial;
- ranking de regiões;
- exportação dos resultados.

<!-- Adicione aqui a imagem da aplicação -->
<img width="1500" alt="Aplicação Streamlit" src="COLE_AQUI_O_LINK_DA_IMAGEM" />

---

# Limitações

Algumas limitações devem ser consideradas:

- o dataset possui informações regionais agregadas, não imóveis individuais;
- existe um limite superior próximo de US$ 500 mil na variável alvo;
- o modelo representa padrões existentes na base histórica;
- as previsões não representam anúncios imobiliários disponíveis em tempo real;
- SHAP e Feature Importance explicam o comportamento do modelo, mas não estabelecem causalidade.

---

# Principais Tecnologias

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
Git / GitHub
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

Crie o ambiente:

```bash
python -m venv .venv
```

Ative no Windows:

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Execute a aplicação:

```bash
python -m streamlit run src/app.py
```

---

# Fluxo do Projeto

```text
Dados
  ↓
Preparação
  ↓
EDA
  ↓
Feature Engineering
  ↓
Modelagem
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
SHAP
  ↓
Market Radar
  ↓
Streamlit
```

---

# Conclusão

O projeto demonstra a construção de uma solução completa de Machine Learning aplicada ao mercado imobiliário.

O modelo final, **XGBoost otimizado com Optuna**, apresentou no conjunto de teste:

```text
R²   = 0.824
MAE  = US$ 30.596
RMSE = US$ 48.129
MAPE = 15.9%
```

A proximidade entre os resultados de Cross-Validation e teste, juntamente com a redução do gap observada na Learning Curve, indica boa estabilidade e capacidade de generalização.

Mais importante do que apenas prever preços, o projeto transforma as estimativas em uma ferramenta de **inteligência geográfica e análise de mercado**, conectando:

```text
Machine Learning
+
Explainable AI
+
Geospatial Analytics
+
Produto de Dados
```

---

# Próximas Evoluções

Como próximos passos:

- integração com dados imobiliários atualizados;
- mapas interativos;
- enriquecimento geoespacial;
- MLflow para rastreamento de experimentos;
- API com FastAPI;
- Docker;
- testes automatizados;
- monitoramento de drift;
- deploy em Cloud;
- integração com anúncios imobiliários reais.

---

# Autor

**Renan Assis Trevelim**

Projeto desenvolvido como aplicação prática de:

**Data Science • Machine Learning • Explainable AI • Geospatial Analytics • Streamlit**