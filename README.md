# Model Card: xRFM

## 1. Detalhes do modelo

- **Nome:** xRFM
- **Versão:** 0.4.5
- **Autores originais:**  Beaglehole et al., 2026
- **Repositório oficial:** https://github.com/dmbeaglehole/xRFM
- **Licença do código:** MIT
- **Família arquitetural:** Baseada em árvores e máquinas de kernel
- **Contagem de parâmetros:** N/A
- **Complexidade computacional:** $O(nlogn)$ em tempo de treinamento, e $O(logn)$
- **Pico de memória observado:** 50% da capacidade da GPU durante o período de otimização do Optuna para o xRFM (dado extraído via nvidia-smi).
- **Toolkit / dependências:** torch 1.8.x,
    numpy 1.19.x,
    scikit-learn 1.4.x,
    tqdm 4.60.x
- **Hiperparâmetros principais:** iters, lambda, gamma, max_leaf_size; a busca pelos melhores hiperparâmetros foi realizado por meio do Optuna via Otimização Bayesiana

## 2. Uso pretendido

- **Caso de uso primário:** classificação supervisionada em dados tabulares.
- **Casos de uso fora de escopo:** N/A
- **Usuários pretendidos:** Pesquisadores e profissionais de ciência de dados.
- **Faixa de n suportada:** de $1.000$ amostras até $10.000$ amostras
- **Faixa de p suportada:** de $6$ a $76$ features
- **Condições operacionais:** Multithreading Nativo de CPU em uma AMD RX 5700X com 32GB de RAM para LightGBM e CatBoost, CUDA em uma RTX 3060Ti para xRFM e XGBoost. Ambiente Virtual configurado para Python 3.11.

## 3. Fatores observados

Dimensões em que o desempenho do modelo varia, avaliadas neste projeto sobre os 9 datasets do TabArena-v0.1:

- **Tamanho do dataset (n):** Para regimes pequenos (< 1000 amostras) o modelo possui a menor AUC OvO, alcançando seu pico de desempenho em regimes médios (1.000 a 10.000 amostras), e diminuindo novamente em datasets grandes (> 10.000 amostras).
- **Número de classes:** O modelo possuiu um melhor desempenho médio em regimes multiclasses do que em regimes binários.
- **Proporção entre features categóricas e numéricas:** Em datasets que possuem maior proporção de features categóricas o modelo performou de forma ligeiramente superior. 
- **Presença de valores ausentes:** O desempenho médio do modelo com valores ausentes é ligeiramente inferior ao seu desempenho quando o dataset não possui valores ausentes. 

## 4. Métricas alcançadas

Tabela agregada nos 9 datasets do TabArena. Reportar média, desvio padrão e intervalo de confiança de 95% via bootstrap (1.000 reamostragens).

| Métrica | Média | Desvio | IC 95% (bootstrap) |
|---|---|---|---|
| AUC OvO | $0,904$ | $0,108$ | <[0,0000; 0,0000]> |
| Accuracy | $0,869$  | $0,099$  | <[0,0000; 0,0000]> |
| G-Mean | $0,741$  | $0,237$  | <[0,0000; 0,0000]> |
| Cross-Entropy | $0,368$  | $0,214$  | <[0,0000; 0,0000]> |
| Tempo total (s) | $9,267$  | $19,259$  | <[0,0; 0,0]> |

### Resultados por regime

- **Tamanho:** pequeno: AUC=$0,792$ ; médio: AUC=$0,992$; grande: AUC=$0,928$
- **Número de classes:** binário: AUC=$0,897$; multiclasse: AUC $\approx 1.000$
- **Proporção categórica:** baixa: AUC=$0,880$; alta: AUC=$0,939$
- **Missing values:** com NaN: AUC=$0,97$; sem NaN: AUC=$0,878$ 

## 5. Dados de avaliação

- **Origem:** 9 datasets do TabArena-v0.1 (NeurIPS 2025), via OpenML.
- **Distribuição por regime:** 3 pequenos + 3 médios + 3 grandes.
- **Estratégia de split:** 70/30 estratificado por classe, seed=42.
- **Pré-processamento aplicado:** A estrategia de imputação utilizada foi baseada na estratégia de imputação da estrutura de dados do Pandas. Os valores númericos ausentes foram tratados por meio do cálculo da mediana no conjunto de treinamento (assumido 0.0 quando em indeterminação matemática)., mantendo as variáveis categóricas estritamente preservadas. A codificação das variáveis categóricas e isolamento de escala foram delegados às fábricas de construção de modelos, com a função de construção do modelo convertendo variáveis categóricas em numéricas, e a escala sendo gerenciada pelas estruturas do pytabkit. 
- **Lista dos datasets utilizados:** 

| Nome | Task ID | Amostras | Features | Classes | Regime |
|---|---|---|---| --- | --- |
|hepatitis|$54$|$155$|$19$| $2$ | pequeno |
|blood-transfusion-service-center|10101|$748$|$4$| $2$ | pequeno |
|diabetes|$37$|$768$|$8$| $2$ | pequeno |
|car|$21$|$1728$|$6$| $4$ | médio |
|mfeat-fourier|$14$|$2000$|$76$| $10$ |médio |
|kr-vs-kp|$3$|$3196$|$36$| $2$ | médio |
|MagicTelescope|$3954$|$19020$|$11$| $2$ | grande |
|letter|$6$|$20000$|$16$| $26$ | grande |
|bank-marketing|$14965$|$45211$|$16$| $2$ | grande |


## 6. Dados de treino e pré-treino

- **Modelo é foundation model pré-treinado, treinado do zero ou híbrido?** Treinado do zero
- **Origem dos dados de pré-treino (se aplicável):** N/A
- **Origem dos dados de treino direto (se aplicável):** Treino direto nos dados de split.
- **Possíveis vieses herdados do pré-treino:** N/A

## 7. Análise quantitativa

- **Posição no ranking médio entre os 4 sistemas avaliados** (modelo do grupo + 3 baselines): 3 de 4
- **Comparação com baselines (LightGBM, CatBoost, XGBoost):** O delta entre o AUC OvO médio (em testes) em relação a cada regime foi: XGBoost=-0.02; CatBoost=-0.018; LightGBM=0.02. Assim, o delta médio de AUC OvO foi de -0.06, sendo inferior aos baselines em todos os datasets observados. 
- **Custo vs. desempenho:** Possui um custo total médio de 9,267s, sendo o segundo pior modelo em quesito de tempo total de inferência no conjunto de teste. Isso se dá pela natureza dessa arquitetura, que  escala de forma muito expressiva com o tamanho do dataset (o maior dataset possui um tempo de execução de 58s, enquanto o menor possui um tempo de 0,016s). 
- **Quebra por regime:** No regime de valores ausentes, o modelo se mostrou competitivo em cenários sem valores ausentes, e estável quando há valores ausentes; podemos associar isso ao kernel do RFM, que trata todos os pontos como vetores no espaço de distância. No regime de tamanho de dataset, o modelo se mostrou sobreajustado, possuindo um gargalo em poucas amostras por sobreajuste, e um gargalo em tempo para muitas amostras por conta das operações com matrizes custosas nas folhas. No regime de proporção categórica, o modelo do grupo se manteve competitivo, mas não ficou conclusivo se o motivo foi por conta do da escolha dos datasets. Por último, no regime de tipo de classificação, o modelo foi muito competitivo em modelos multiclasse, com estrutura de decisão separável no espaço de features, mas em classificações binárias teve uma alta variabilidade, principalmente pelo sobreajustamento do modelo

## 8. Avisos e recomendações

- **Quando usar este modelo:** Em regimes com um número de amostras entre $1.000$ a $10.000$ amostras com mais de 10 features, onde o modelo terá um desempenho alto com grande rapidez em treino e em teste. 
- **Quando NÃO usar este modelo:** Em regimes com poucas features, ou com muitas amostras (> $10.000$), pois correm o risco de se sobreajustar ou de demorar muito tempo nas folhas, respectivamente.
- **Alternativas recomendadas em cada caso:** Em datasets grandes, utilizar o CatBoost. Em datasets pequenos, utilizar o LightGBM

## 9. Considerações éticas

- **Riscos de uso indevido:** Quando usado em situação indevida, pode ter um desempenho muito baixo, favorecer a classe majoritária utilizada em treino, ou demandar um alto custo computacional.
- **Fairness por classe:** Em cenários de classificação binária desbalanceada, o modelo tende a sobreajustar para a classe majoritária, de forma que o uso nesse tipo de situação não é aconselhável.
- **Impacto ambiental:** Em datasets pequenos e médios, a latência de inferência é baixa ou competitiva em relação a outros modelos, não sendo um problema ambiental em gasto energético. Em datasets grandes, contudo, o grande tempo de inferência pelas operações matemáticas complexas envolvidas pode ser um sério problema, se o modelo for usado nesse tipo de contexto.
- **Recomendações de auditoria:** Os AGOPs das folhas podem ser usados para descobrir as features de maior importância para a classificação, adicionando uma explicabilidade maior ao modelo que pode ser importante em situações de auditoria

## 10. Reprodutibilidade

- **Ambiente:** Python <3.11>, dependências fixadas em `pyproject.toml`.
- **Hardware utilizado:** <CPU, RAM, tempo total de execução>
- **Comandos para reproduzir:**
  ```bash
  uv sync
  python -m src.pipeline.run_all --include-group-model --seed 42
  ```
- **Hash do commit:** 93b636cb43064b214bee723aa0f9ee9b96081695 [https://github.com/boahorapedro/projeto-final-am/commit/93b636cb43064b214bee723aa0f9ee9b96081695]

## Referências

- Beaglehole, D. et al. (2025). Recursive Feature Machine (xRFM). Repositório oficial: https://github.com/dmbeaglehole/xRFM
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*.
- TabArena-v0.1 (NeurIPS 2025): https://tabarena.ai
