# Projeto Final de Aprendizagem de Máquina, Repositório-Template

Estrutura de código de referência para a Etapa 2 do Projeto Final da disciplina de Aprendizagem de Máquina. O template padroniza:

1. Carregamento de 9 datasets do TabArena-v0.1 (3 small + 3 medium + 3 large).
2. Implementação dos baselines (LightGBM, CatBoost, XGBoost) via `pytabkit`.
3. Pipeline de split (70/30 com seed fixa), tuning com Optuna, avaliação das métricas exigidas.
4. Análise descritiva: tabela média ± desvio por modelo, gráficos de barras e boxplot, análise por regime.
5. Smoke test que valida os 3 baselines mais o pipeline de gráficos em um dataset pequeno.

## Status do template

A tabela abaixo indica, para cada componente, o que já está implementado e validado e o que cabe a cada grupo preencher. As convenções são:

- **Pronto:** componente implementado e validado por smoke test; nada a fazer.
- **Esqueleto:** há um arquivo funcional, porém genérico, que cada grupo deve adaptar ao seu modelo.
- **Placeholder:** há apenas um stub; o grupo precisa implementar.
- **Ação do aluno:** entrega que não está no template e o grupo deve produzir do zero.

| Componente | Status | O que falta |
|---|---|---|
| `data/load_tabarena.py`, carregamento via OpenML | Pronto | Substituir `RECOMMENDED_TASK_IDS` (lista provisória de 9 IDs) pela lista oficial escolhida, validando com `summarize()`. |
| `src/models/baselines.py` (LightGBM, XGBoost, CatBoost via pytabkit) | Pronto | Nada. |
| `src/models/group_model.py` (modelo principal do grupo) | Placeholder | Implementar `build_group_model(seed)` retornando o estimador atribuído (ver exemplos comentados no próprio arquivo). |
| `src/pipeline/split.py` (70/30 estratificado) | Pronto | Nada. |
| `src/pipeline/tune.py` (Optuna, genérico) | Esqueleto | Cada grupo define o `search_space` apropriado para o seu modelo. Para baselines `pytabkit` com defaults TD, o tuning pode ser opcional. |
| `src/pipeline/evaluate.py` (AUC OvO, ACC, G-Mean, CE, tempo) | Pronto | Nada. |
| `src/pipeline/regime.py` (quebra por regime) | Pronto | Nada. |
| `src/pipeline/run_all.py` (orquestrador CLI) | Pronto | Cada grupo passa `--include-group-model` após implementar `group_model.py`. |
| `src/reports/results_table.py` (resumos e exportação Markdown) | Pronto | Nada. |
| `src/reports/plots.py` (barras, boxplot, regime, time vs. quality) | Pronto | Nada. |
| `notebooks/01_eda.ipynb` até `04_analise_resultados.ipynb` | Esqueleto | Executar e adaptar; usar para gerar figuras e tabelas do relatório. |
| `model_cards/TEMPLATE.md` | Placeholder | Copiar para `model_cards/<seu-modelo>.md` e preencher as 10 seções (detalhes do modelo, uso pretendido, fatores, métricas com IC 95%, dados de avaliação, dados de treino e pré-treino, análise quantitativa, avisos e recomendações, considerações éticas, reprodutibilidade). |
| `tests/test_pipeline.py` (smoke test) | Pronto | Nada. |
| Tabela de seleção dos 9 datasets (no relatório) | Ação do aluno | Construir tabela com nome, task ID OpenML, n, n_features, n_classes e regime. |
| Relatório final em PDF | Esqueleto | Estrutura sugerida em `entregaveis/relatorio-template.pdf`. Substituir placeholders e preencher conforme as exigências da disciplina. |
| Slides da apresentação (10 minutos: 5 + 5) | Esqueleto | Estrutura sugerida em `entregaveis/slides-template.pdf` (cerca de 12 frames, dois blocos de 5 minutos). |
| Rubrica de avaliação | Pronto | Disponível em `entregaveis/rubrica.pdf`. Vincula a nota a entregas concretas, decompondo o esquema 40 + 50 + 10 do PDF da disciplina em critérios com pesos específicos. |

## Modelos atribuíveis aos grupos

| # | Modelo | Toolkit |
|---|---|---|
| 1 | TabPFN-2.5 | `tabpfn` |
| 2 | TabICL v2 | `tabicl` |
| 3 | TabM | `pytabkit` |
| 4 | ModernNCA | `LAMDA-Tabular/TALENT` |
| 5 | RealMLP | `pytabkit` |
| 6 | xRFM | `xrfm` |
| 7 | Mambular | `deeptab` |
| 8 | FT-Transformer | `pytabkit` ou `deeptab` |
| 9 | EBM | `interpret` |
| 10 | Trompt | `pytorch-frame` |

## Quickstart

Pré-requisitos: Python 3.11 ou superior e [`uv`](https://docs.astral.sh/uv/) (recomendado) ou `pip`.

```bash
# clonar o repositório
git clone <url-do-template>
cd projeto-final-AM-grad

# opção A: uv (recomendado)
uv sync

# opção B: pip
pip install -e .

# rodar o smoke test (verifica que cada baseline e os gráficos executam)
pytest tests/test_pipeline.py -v
```

Para rodar o experimento completo com o seu modelo atribuído:

```bash
# editar src/models/group_model.py para apontar para o modelo do grupo
# rodar o pipeline em todos os 9 datasets:
python -m src.pipeline.run_all --include-group-model --seed 42
```

## Estrutura

```
projeto-final-AM-grad/
|- README.md
|- pyproject.toml
|- Dockerfile
|- .python-version
|- data/
|   |- load_tabarena.py
|- src/
|   |- models/
|   |   |- baselines.py
|   |   |- group_model.py
|   |- pipeline/
|       |- split.py
|       |- tune.py
|       |- evaluate.py
|       |- regime.py
|       |- run_all.py
|   |- reports/
|       |- results_table.py
|       |- plots.py
|- notebooks/
|   |- 01_eda.ipynb
|   |- 02_demo_baselines.ipynb
|   |- 03_demo_modelo_grupo.ipynb
|   |- 04_analise_resultados.ipynb
|- model_cards/
|   |- TEMPLATE.md
|- docs/
|   |- INFRAESTRUTURA.md
|- tests/
    |- test_pipeline.py
```

## Fluxo de trabalho recomendado

1. **EDA inicial:** rodar `notebooks/01_eda.ipynb` para inspecionar os 9 datasets.
2. **Baselines:** rodar `notebooks/02_demo_baselines.ipynb` para confirmar que LightGBM, CatBoost e XGBoost executam end-to-end.
3. **Modelo do grupo:** implementar o wrapper em `src/models/group_model.py` e validar em `notebooks/03_demo_modelo_grupo.ipynb`.
4. **Experimento completo:** rodar `python -m src.pipeline.run_all --include-group-model --seed 42`.
5. **Análise descritiva e por regime:** abrir `notebooks/04_analise_resultados.ipynb` e gerar tabela, barras, boxplot e gráficos por regime.
6. **Model card:** copiar `model_cards/TEMPLATE.md` para `model_cards/<nome-do-modelo>.md` e preencher.

## Reprodutibilidade

1. Seed fixa em todas as etapas (`split`, `tune`, `evaluate`).
2. Versões fixadas no `pyproject.toml`.
3. Dockerfile opcional disponível para containerização.
4. Saídas intermediárias (resultados por dataset por modelo) são gravadas em CSV em `results/`.

## Licenças

Todas as bibliotecas utilizadas têm licenças permissivas (MIT, Apache 2.0). A única exceção é o modelo TabPFN-2.5, cujos pesos são distribuídos sob licença não-comercial; o uso acadêmico em sala está explicitamente autorizado.

## Suporte

Em caso de dúvida técnica, consulte primeiro `docs/INFRAESTRUTURA.md`. Se a dúvida persistir, abrir issue no repositório do template ou contatar o professor da disciplina.
