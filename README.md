# Estimativa de BCS (Body Condition Score)

Modelo de regressão para estimar o **Body Condition Score** (escala de
condição corporal, 3.25 a 4.25 em passos de 0.25) de vacas leiteiras a
partir de uma única foto da região da garupa, usando transfer learning
com MobileNetV2.

## Contexto

BCS é uma métrica usada na pecuária leiteira pra avaliar reservas de
gordura corporal do animal — importante pra saúde reprodutiva e
produção de leite, tradicionalmente avaliada por inspeção visual e
tátil de um especialista. Este projeto testa se um modelo de visão
computacional, usando só uma fotografia, consegue aproximar essa
avaliação.

## Dataset

- ~53.500 imagens, 5.704 vacas, 5 classes de BCS (3.25, 3.5, 3.75, 4.0, 4.25)
- Anotação no formato Pascal VOC (XML), com bounding box do animal
- Imagens já enquadradas na região da garupa — sem recorte adicional necessário
- Em média, ~9-10 imagens por vaca (frames de vídeo, não fotos independentes)

### O risco de vazamento de dado

O dataset mistura duas convenções de nome de arquivo:

- **`GS_1160_2.jpg`** (prefixo + ID da vaca + número do frame) — permite
  agrupar com segurança todos os frames da mesma vaca
- **`L-i5494.jpg`** (prefixo + índice sequencial) — não contém ID de
  vaca explícito

Com ~9-10 imagens por vaca, um split aleatório por imagem correria o
risco de colocar frames quase idênticos da mesma vaca em treino e
em validação/teste — inflando artificialmente a métrica de validação
sem generalização real.

### A tentativa de correção

A primeira hipótese pra agrupar os arquivos do segundo padrão foi
clustering por gap: números em sequência com saltos pequenos
seriam a mesma vaca; saltos grandes marcariam troca de vaca. A
distribuição real de gaps (94% dos gaps = 1, indicando bom sinal)
sustentou essa hipótese inicialmente.

A validação, porém, revelou um grupo de 377 imagens com o valor de BCS
alternando entre 3.75, 4.00 e 4.25 dentro do mesmo grupo — algo
fisiologicamente impossível para uma única vaca. Isso provou que a
numeração sequencial não separa vacas de forma confiável em todos os
casos: vacas diferentes, por coincidência, tinham numeração próxima o
suficiente para escapar do limiar de corte.

### A solução final

Tratamento assimétrico do dado, dependendo da confiabilidade do
agrupamento:

| Origem do dado | Uso |
|---|---|
| ID de vaca explícito no nome (`reliable_group = True`) | Treino, validação e teste |
| Sem ID confiável (`reliable_group = False`) | Só treino |

Isso preserva 100% do volume de dado disponível, sem sacrificar a
integridade da avaliação — validação e teste são construídos
exclusivamente sobre dado com garantia de não-vazamento.

## Enquadramento do problema

**Regressão**, não classificação — apesar do dataset ter só 5 valores
discretos de rótulo. BCS é uma escala ordinal contínua por natureza
(avaliadores humanos atribuem valores como 3.25 ou 3.5), e regressão
captura essa ordem naturalmente: um erro entre classes vizinhas é
penalizado como menor que um erro entre classes extremas, o que não
aconteceria com classificação categórica.

## Arquitetura e treino

- **Base:** MobileNetV2 pré-treinado (ImageNet), `include_top=False`
- **Cabeça:** `GlobalAveragePooling2D → Dropout → Dense(64) → Dropout → Dense(1, linear)`
- **Augmentation:** flip horizontal, brilho, contraste, zoom (embutida no modelo)
- **Treino em duas fases:**
  1. Base congelada, só a cabeça treina (LR 1e-3)
  2. Fine-tuning dos ~30% finais da base (LR 1e-5, 100x menor, para não destruir os pesos pré-treinados)
- **Loss:** Huber (robusta a outliers/rótulos ambíguos)
- **Split:** 70% treino / 15% validação / 15% teste, por vaca (ver seção acima)

Notebook: [`01_treino_bcs.ipynb`](01_treino_bcs.ipynb)

## Resultados

| Métrica | Valor |
|---|---|
| MAE (teste) | 0.207 pontos de BCS |
| Acerto exato (±0.125) | 35.0% |
| Acerto com tolerância de 1 classe (±0.25) | 66.4% |

O modelo aprende sinal real da imagem (supera a baseline ingênua), mas
apresenta um viés claro: erro nas classes extremas (3.25 e 4.25) é
cerca do dobro do erro nas classes centrais — o modelo tende a
"comprimir" suas predições em direção à classe majoritária (3.75, a
mais frequente no dataset).

### Comparação com a literatura

| Sistema | Tipo de captura | Acerto em ±0.25 |
|---|---|---|
| Este projeto | Imagem 2D única | 66.4% |
| Sistema automatizado, JDS (2023) | Imagem 2D, >34 mil anotações veterinárias | 84.6% |
| Zhao et al. (2023) | Imagem de profundidade + EfficientNet | 91.2% |
| Shi et al. (2023) | Nuvem de pontos 3D + atenção | 80.0% |

O resultado deste projeto está abaixo dos sistemas publicados, mas
usando uma abordagem estruturalmente mais simples (imagem 2D única,
regressão direta, sem profundidade/3D) e um volume de anotação bem
menor. A diferença é consistente com essas limitações, não um
indicativo de erro de implementação.

## Correção de desbalanceamento (retreino)

Duas estratégias de peso por amostra foram testadas pra corrigir o
viés nas classes extremas — detalhes e saídas completas em
[`02_retreino_ponderado.ipynb`](02_retreino_ponderado.ipynb).

| Classe | Original | v2 (Huber + peso linear) | v3 (MSE + peso²) |
|---|---|---|---|
| 3.25 | 0.330 | 0.306 | **0.294** |
| 3.50 | 0.161 | **0.152** | 0.177 |
| 3.75 | **0.141** | 0.154 | 0.179 |
| 4.00 | **0.187** | 0.200 | 0.190 |
| 4.25 | 0.338 | 0.325 | **0.300** |
| MAE geral | 0.207 | **0.206** | 0.211 |
| Tolerância ±0.25 | 66.4% | **67.0%** | 66.3% |

**Conclusão:** as duas correções reduzem o erro nas classes extremas,
mas pioram as classes centrais — uma redistribuição de erro, não
uma redução real. O padrão se repete de forma consistente em duas
estratégias diferentes (loss e intensidade de peso distintas), o que é
evidência mais forte do que um experimento isolado teria dado: o fator
limitante provavelmente é volume de dado nas classes extremas, não
a estratégia de otimização escolhida. Corrigir isso exigiria mais dado
anotado nessas classes, não um ajuste adicional de hiperparâmetro.

## Limitações conhecidas

- ~17% do dataset (imagens sem ID de vaca confiável) participa só do
  treino — não há garantia formal de que essas imagens não têm alguma
  forma de correlação entre si não capturada pelo tratamento adotado
- O desbalanceamento de classe não foi resolvido, só parcialmente
  mitigado — classes extremas continuam com erro consideravelmente
  maior que as centrais
- Comparação com a literatura é aproximada, não algo
  controlado nas mesmas condições (dataset, protocolo de anotação, e
  espécie/raça diferentes de estudo para estudo)

## Próximos passos

- Testar reamostragem física das classes extremas, como
  mecanismo diferente de correção de desbalanceamento (peso na loss vs.
  composição real do batch)
- Explorar regressão ordinal como alternativa à regressão linear simples
- Avaliar se captura com múltiplas imagens/ângulos por avaliação (em
  vez de foto única) reduz o gap em relação aos sistemas publicados



