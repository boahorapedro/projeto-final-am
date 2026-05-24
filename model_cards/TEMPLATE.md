# Model Card: <Nome do Modelo>

> Preencha este template para o modelo principal atribuído ao seu grupo. Substitua os campos `<...>` pelos valores reais. Não deixe campos em branco; use "N/A" quando não aplicável.
>
> Estrutura inspirada em Mitchell et al. (2019), com extensões específicas da disciplina (fatores observados nos quatro regimes do TabArena, análise quantitativa contra os três baselines, e seção de avisos e recomendações).

## 1. Detalhes do modelo

- **Nome:** <ex.: TabM>
- **Versão:** <ex.: 1.0>
- **Autores originais:** <ex.: Gorishniy et al., 2025>
- **Repositório oficial:** <URL>
- **Licença do código:** <ex.: Apache 2.0>
- **Família arquitetural:** <ex.: ensemble parameter-efficient de MLPs>
- **Contagem de parâmetros:** <ex.: 1,2M; reportar treináveis vs. fixos quando aplicável>
- **Complexidade computacional:** <tempo e memória em função de n e p; ex.: O(np) em treino e em inferência>
- **Pico de memória observado:** <ex.: 4 GB em datasets do regime grande>
- **Toolkit / dependências:** <ex.: pytabkit 1.x, pytorch 2.x>
- **Hiperparâmetros principais:** <listar; indicar se foi feita busca via Optuna>

## 2. Uso pretendido

- **Caso de uso primário:** classificação supervisionada em dados tabulares.
- **Casos de uso fora de escopo:** <ex.: dados não-IID, séries temporais, dados de imagem>
- **Usuários pretendidos:** <ex.: alunos e praticantes em problemas tabulares com benchmarks padronizados>
- **Faixa de n suportada:** <ex.: até 50.000 amostras com bom desempenho>
- **Faixa de p suportada:** <ex.: até 1.000 features após codificação>
- **Condições operacionais:** <ex.: roda em laptop com 8 GB de RAM; GPU opcional>

## 3. Fatores observados

Dimensões em que o desempenho do modelo varia, avaliadas neste projeto sobre os 9 datasets do TabArena-v0.1:

- **Tamanho do dataset (n):** <descrever sensibilidade do modelo: pequeno (< 1.000), médio (1.000 a 10.000), grande (> 10.000)>
- **Número de classes:** <binário vs. multiclasse>
- **Proporção entre features categóricas e numéricas:** <baixa vs. alta>
- **Presença de valores ausentes:** <com NaN vs. sem NaN; estratégia de imputação adotada>

## 4. Métricas alcançadas

Tabela agregada nos 9 datasets do TabArena. Reportar média, desvio padrão e intervalo de confiança de 95% via bootstrap (1.000 reamostragens).

| Métrica | Média | Desvio | IC 95% (bootstrap) |
|---|---|---|---|
| AUC OvO | <0,0000> | <0,0000> | <[0,0000; 0,0000]> |
| Accuracy | <0,0000> | <0,0000> | <[0,0000; 0,0000]> |
| G-Mean | <0,0000> | <0,0000> | <[0,0000; 0,0000]> |
| Cross-Entropy | <0,0000> | <0,0000> | <[0,0000; 0,0000]> |
| Tempo total (s) | <0,0> | <0,0> | <[0,0; 0,0]> |

### Resultados por regime

- **Tamanho:** pequeno: AUC=<...>; médio: AUC=<...>; grande: AUC=<...>
- **Número de classes:** binário: AUC=<...>; multiclasse: AUC=<...>
- **Proporção categórica:** baixa: AUC=<...>; alta: AUC=<...>
- **Missing values:** com NaN: AUC=<...>; sem NaN: AUC=<...>

## 5. Dados de avaliação

- **Origem:** 9 datasets do TabArena-v0.1 (NeurIPS 2025), via OpenML.
- **Distribuição por regime:** 3 pequenos + 3 médios + 3 grandes.
- **Estratégia de split:** 70/30 estratificado por classe, seed=<n>.
- **Pré-processamento aplicado:** <descrever imputação, codificação categórica e escalonamento>.
- **Lista dos datasets utilizados:** <preencher com nome, OpenML task ID, n, n_features, n_classes, regime; ver tabela do relatório>.

## 6. Dados de treino e pré-treino

- **Modelo é foundation model pré-treinado, treinado do zero ou híbrido?** <responder>
- **Origem dos dados de pré-treino (se aplicável):** <ex.: 130M de datasets sintéticos gerados pelos autores>
- **Origem dos dados de treino direto (se aplicável):** <ex.: pesos inicializados aleatoriamente; treino direto nos splits do projeto>
- **Possíveis vieses herdados do pré-treino:** <descrever; relevante para foundation models>

## 7. Análise quantitativa

- **Posição no ranking médio entre os 4 sistemas avaliados** (modelo do grupo + 3 baselines): <x de 4>
- **Comparação com baselines (LightGBM, CatBoost, XGBoost):** <delta médio em AUC OvO; em quais datasets o modelo do grupo vence; em quais perde>
- **Custo vs. desempenho:** <tempo total do modelo do grupo vs. baselines; trade-off observado>
- **Quebra por regime:** <em quais regimes o modelo do grupo vence; em quais perde; provável explicação alinhada à arquitetura>

## 8. Avisos e recomendações

- **Quando usar este modelo:** <regimes em que mostrou melhor desempenho ou melhor custo-benefício>
- **Quando NÃO usar este modelo:** <regimes onde os baselines vencem ou onde restrições operacionais inviabilizam o uso>
- **Alternativas recomendadas em cada caso:** <ex.: para datasets pequenos, usar LightGBM TD; para datasets com cardinalidade categórica alta, usar CatBoost TD>

## 9. Considerações éticas

- **Riscos de uso indevido:** <ex.: viés herdado dos dados de treino, decisões opacas em domínios sensíveis>
- **Fairness por classe:** <recall e precisão por classe; classes minoritárias com baixo recall>
- **Impacto ambiental:** <energia consumida durante tuning; latência de inferência>
- **Recomendações de auditoria:** <ex.: comparar predições com baseline interpretável como EBM antes de deploy>

## 10. Reprodutibilidade

- **Ambiente:** Python <3.11>, dependências fixadas em `pyproject.toml`.
- **Hardware utilizado:** <CPU, RAM, tempo total de execução>
- **Comandos para reproduzir:**
  ```bash
  uv sync
  python -m src.pipeline.run_all --include-group-model --seed 42
  ```
- **Hash do commit:** <git rev-parse HEAD>

## Referências

- <citação do paper original do modelo>
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*.
- TabArena-v0.1 (NeurIPS 2025): https://tabarena.ai
