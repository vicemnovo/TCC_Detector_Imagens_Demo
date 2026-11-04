# Demonstração do detector de imagens — Fase 3.9

Este é um repositório independente de apresentação. O notebook usa o melhor
checkpoint disponível da Fase 3.9 no **Google Drive do grupo** para estimar a
chance de uma imagem ser sintética. Nenhum checkpoint, imagem, dataset ou
credencial é versionado aqui.

## Uso no Google Colab

1. Abra `demo_fase3_9.ipynb` no **navegador do Colab** usando a conta com
   acesso ao Drive do TCC. Conecte uma GPU, se disponível.
2. Execute as células 1 a 5. Elas montam o Drive, copiam o checkpoint para
   `/content`, montam a arquitetura e verificam o carregamento estrito dos
   pesos. Não treinam o modelo.
3. Execute a célula 6 para escolher uma ou mais imagens no navegador. Para
   uso pelo VS Code, coloque imagens na pasta de entrada do Drive criada pela
   célula 7 e execute-a. As imagens são usadas somente para inferência.

O notebook prefere `phase3_final.pt` **somente quando**
`training_complete.json` existe. Nesse caso usa a temperatura e o limiar
calibrados internamente. Se o treino ainda estiver em andamento, usa
`best.pt`, escolhido pela validação interna; o número apresentado será um
*softmax ainda sem calibração* e não haverá classificação por limiar. Se
nenhum desses checkpoints existir, o notebook para com uma mensagem clara.

O percentual mostrado é uma **estimativa do modelo**, não uma prova de
autoria, edição ou veracidade da cena. A meta de 85% de acurácia externa
continua experimental; esta demonstração não mede acurácia nem altera os
conjuntos de avaliação. O código de inferência usa cinco recortes de 224 px,
faz a média dos logits e aplica a temperatura do checkpoint final, como o
protocolo de avaliação da Fase 3.9.

Este repositório tem histórico Git próprio. Os dados de treino, os resultados
anteriores e o repositório do TCC permanecem separados.
