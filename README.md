# Detecção de fraude em cartões

Experimento de classificação com regressão logística, Random Forest, XGBoost e explicações SHAP. O notebook inclui as saídas da execução, com tabelas, curvas e uma explicação de transação individual.

- [Notebook executado](./fraude_cartoes.ipynb)
- [Abrir no Google Colab](https://colab.research.google.com/github/VinnyFigTerminal/deteccao-fraude-cartoes/blob/main/fraude_cartoes.ipynb)
- [Base pública Credit Card Fraud Detection — ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Problema e dados

A base reúne 284.807 transações, com 492 fraudes (0,1727%). Um classificador que marca todas como normais acerta cerca de 99,83% e tem recall zero para fraude. Por isso, o projeto avalia **recall** (fraudes reais encontradas), **precisão** (parte dos alertas que eram fraudes) e **F1** (média harmônica entre as duas). A curva precisão–recall mostra como a capacidade de detectar fraudes varia com a proporção de alertas corretos.

`Time` indica tempo relativo, `Amount` é o valor da transação, `Class` é o rótulo e `V1` a `V28` são componentes PCA. O notebook baixa o ZIP público pelo link da base. O CSV fica fora do repositório.

## Preparação e avaliação

- Divisão estratificada em **60% treino, 20% validação e 20% teste**, com semente 42.
- `np.log1p` em `Amount` e `StandardScaler` dentro de `Pipeline`. O scaler é ajustado em cada treino, inclusive dentro do `GridSearchCV`.
- Random Forest com `class_weight='balanced'` e XGBoost com `scale_pos_weight`. A regressão logística sem ponderação é a linha de base aprendida.
- Busca em grade sobre profundidade e taxa de aprendizado do XGBoost com três dobras estratificadas e **recall** como critério. O modelo final foi escolhido pela **AP na validação**. O limiar foi escolhido pelo **F1 na validação**.

## Comparação no teste, limiar 0,50

| Modelo | Recall fraude | Precisão fraude | F1 fraude | AP |
|---|---:|---:|---:|---:|
| Sempre normal | 0,0000 | 0,0000 | 0,0000 | 0,0017 |
| Regressão logística | 0,6429 | 0,8289 | 0,7241 | 0,7420 |
| Random Forest | 0,7755 | 0,8444 | 0,8085 | 0,8343 |
| XGBoost | 0,8571 | 0,5122 | 0,6412 | 0,8330 |
| XGBoost ajustado por recall | 0,8980 | 0,1337 | 0,2328 | 0,7155 |

O **XGBoost** original ganhou pela AP na validação. No teste, o Random Forest apresentou o maior F1 em 0,50. O XGBoost ajustado encontrou mais fraudes, mas gerou 570 falsos alertas, mostrando o custo de otimizar recall isoladamente.

## Limiar e explicação SHAP

A validação selecionou **0,50** para o XGBoost. No teste, esse limiar identificou **84 de 98 fraudes**, com **80 falsos positivos**. O limiar **0,30** encontrou **88 de 98**, mas elevou os falsos positivos para **168** e reduziu o F1 de **0,6412** para **0,4972**.

Na amostra de 400 transações do teste, **V4, V8 e V14** tiveram as maiores contribuições globais médias em valor absoluto. No alerta analisado, com classe real igual a 1 e escore de aproximadamente **0,99977**, **V14 e V12** foram os maiores fatores que elevaram a saída do modelo. As contribuições estão na escala log-odds. Elas descrevem o comportamento do modelo sobre componentes PCA, sem revelar os atributos originais ou provar relação causal.

## Diferenças em relação ao roteiro da Expert

Além do roteiro proposto, usei uma validação separada para decidir o modelo e o limiar antes de consultar o teste. Incluí um classificador constante, a prevalência como referência na curva precisão–recall, contagens de falsos alertas e uma explicação SHAP global e local. O teste mostrou o custo de baixar o limiar, e a escolha registrada pela validação permaneceu em 0,50.

## Reproduzir

Abra o notebook no Colab e use **Ambiente de execução → Executar tudo**. A primeira célula instala as versões das bibliotecas usadas. Depois do download, o ZIP é reutilizado em execuções seguintes no mesmo ambiente. As células também foram executadas em sequência em Python/IPython, com as saídas registradas no arquivo.

O estudo usa divisão aleatória estratificada e serve como demonstração. Uma validação temporal é necessária para estimar a transferência dos resultados a períodos futuros.
