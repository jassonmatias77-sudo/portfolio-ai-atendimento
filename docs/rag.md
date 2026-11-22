# RAG — Retrieval-Augmented Generation

RAG significa geração aumentada por recuperação. O modelo consulta uma base de
conhecimento antes de responder, em vez de depender apenas do que memorizou
durante o treino.

## O problema que resolve

Um modelo de linguagem sabe o que estava nos dados de treino. Não sabe:

- o conteúdo da documentação interna de uma empresa
- o que mudou depois do corte de treino
- factos específicos de um projecto

Sem RAG, o modelo inventa (alucina) quando lhe perguntam sobre isso.

## Como funciona

1. **Indexação** — os documentos são divididos em pedaços (chunks), cada pedaço
   é convertido num vector (embedding) e guardado.
2. **Recuperação** — a pergunta também vira vector; comparam-se os vectores por
   similaridade de cosseno e escolhem-se os mais próximos.
3. **Geração** — os pedaços recuperados entram no prompt, e o modelo responde
   com base neles.

## Embeddings

Um embedding é uma lista de números que representa o significado de um texto.
Textos com significado parecido têm vectores próximos.

Este projecto usa `nomic-embed-text`, que produz 768 dimensões e corre
localmente em cerca de 50 milissegundos por texto.

## Similaridade de cosseno

Mede o ângulo entre dois vectores, não a distância. Vai de -1 a 1:

- 1 significa mesma direcção (textos equivalentes)
- 0 significa perpendiculares (sem relação)
- -1 significa direcções opostas

A vantagem sobre a distância euclidiana é ser independente do comprimento do
texto.

## Chunking

Dividir documentos demasiado grandes é necessário porque:

- o contexto do modelo é limitado
- um chunk pequeno recupera informação mais precisa

A sobreposição entre chunks evita cortar uma ideia ao meio. Se uma frase
atravessa a fronteira, aparece inteira em pelo menos um dos chunks.

## Limitações

- A busca por força bruta compara com todos os vectores. Acima de cerca de
  10 000 documentos é preciso um índice aproximado como HNSW.
- Sem re-ranking, os resultados podem incluir chunks pouco relevantes.
- Se o documento não contém a resposta, o RAG não a cria — e o modelo deve
  dizer que não sabe.
