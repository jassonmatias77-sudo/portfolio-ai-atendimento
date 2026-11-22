# Latência em pipelines de IA

Latência é o tempo entre a pergunta e a primeira resposta útil. É a métrica
que mais afecta a percepção de qualidade.

## Onde o tempo se vai

Num pipeline com RAG, o tempo divide-se em três partes:

1. **Embedding da pergunta** — cerca de 50 ms localmente
2. **Recuperação** — depende do número de documentos; força bruta é O(n)
3. **Geração** — o modelo a produzir texto, proporcional ao número de tokens

Em modelos locais, há ainda um custo escondido: **carregar o modelo na
memória**. A primeira chamada depois de um reinício pode levar 2 a 5 segundos.

## Como reduzir

**keep_alive** — manter o modelo carregado entre chamadas. No Ollama,
`keep_alive: "30m"` evita o recarregamento. Foi a diferença entre 2,1 s e
0,4 s por pergunta neste projecto.

**Streaming** — devolver os tokens à medida que são gerados. O tempo total não
muda, mas o tempo até ao primeiro token cai de segundos para centenas de
milissegundos. Para voz, é a diferença entre parecer vivo e parecer travado.

**Cache** — perguntas repetidas podem devolver resposta guardada. Útil quando
o padrão de uso é repetitivo.

**Modelo menor** — um modelo de 7B responde em menos tempo que um de 30B. Para
tarefas simples, a diferença de qualidade é menor do que a de velocidade.

## Medir antes de optimizar

Sem medir por etapa, optimiza-se o sítio errado. Neste projecto, a primeira
medição mostrava 2,1 s totais. Assumir que o LLM era o gargalo estaria errado:
0,3 s eram o carregamento do modelo, resolvido com `keep_alive`.

A telemetria por etapa é o que permite essa distinção.
