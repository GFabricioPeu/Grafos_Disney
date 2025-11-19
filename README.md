# 🎬 Sistema de Recomendação de Filmes/Séries

Este projeto é um **Sistema de Recomendação** que utiliza técnicas de Processamento de Linguagem Natural (NLP) e Teoria dos Grafos para sugerir filmes e séries. As recomendações são baseadas na **similaridade semântica** das descrições e na **similaridade de conexões** no grafo de relacionamento entre filmes, atores, diretores, países e categorias.

O trabalho foi desenvolvido para a disciplina de **Teoria dos Grafos** da Universidade Federal da Grande Dourados (UFGD).

---

## ⚙️ Tecnologias e Bibliotecas

O projeto é desenvolvido em Python e requer as seguintes bibliotecas:

* **`pandas`**: Para manipulação e tratamento dos dados do catálogo.
* **`numpy`**: Para operações numéricas de alto desempenho.
* **`networkx`**: Para a construção e análise do grafo de relacionamentos.
* **`scikit-learn`**: Para as técnicas de NLP e *clustering* (TF-IDF e MiniBatchKMeans).
* **`matplotlib`**: Para visualização (exemplo: visualização do subgrafo de recomendações).

---

## 🚀 Como Rodar o Projeto

### Pré-requisitos

Certifique-se de ter o **Python 3** instalado.

Instale as bibliotecas necessárias via `pip`:

```bash
pip install pandas numpy networkx scikit-learn matplotlib