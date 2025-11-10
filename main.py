import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import sklearn.feature_extraction.text as skt
import sklearn.cluster as skc
from sklearn.metrics.pairwise import cosine_similarity
import re

remover = {'as','the', 'and', 'if','that','in','when','so','where','at','but','or','to'}



def limpa_description(D):
    if not isinstance(D,str):
        return ""

    palavras = D.split()
    palavras_resultado = [palavra for palavra in palavras if palavra.lower() not in remover]

    return ' '.join(palavras_resultado)

def limpa_lista(C):
    if not isinstance(C,str):
        return []
    
    itens =[itens.strip() for item in C.split (',') if item.strip()]

    return itens


def trata_arquivo (corpus):
    
    corpus['description']= corpus['description'].apply(limpa_description)
    colunas = ['cast','director','listed_in','country']
    for col in colunas:
        corpus[col] = corpus[col].apply(limpa_lista)
    
    return corpus



def cria_grafo (arquivo):
    df = pd.read_csv(arquivo)
    corpus = df[['show_id','type','title','director','cast','country','rating','listed_in','description']].fillna('')
    
    corpus = trata_arquivo(corpus)
   
    vectorizer = skt.TfidfVectorizer()
    matriz_vetorizada = vectorizer.fit_transform(corpus['description'])

    kmeans = skc.MiniBatchKMeans(n_clusters=100,random_state=9, n_init=10).fit(matriz_vetorizada)
    corpus['cluster'] = kmeans.labels_

    grafo = nx.Graph()

    corpus_index = corpus.set_index('show_id')
    for show_id, row in corpus_index.iterrows():
        grafo.add_node(
            show_id, 
            title=row.get('title', 'N/A'), 
            type=row.get('type', 'N/A'),
            cluster=row.get('cluster', -1),
            cast=row.get('cast',[])
        )


    similaridade = cosine_similarity(matriz_vetorizada)
    lim_similaridade = 0.1

    show_ids = corpus['show_id'].values

    for i in range(similaridade.shape[0]):
        for j in range(i+1,similaridade.shape[1]):
            if similaridade[i,j] > lim_similaridade:
                grafo.add_edge(corpus.iloc[i]['show_id'],corpus.iloc[j]['show_id'],weight=similaridade[i,j])

    return grafo

def mostra_vizinhos(grafo,id_central):

    if id_central not in grafo: #Ve se o nó realmente ta no grafo
        print("No nao esta no grafo")
        return
    
    vizinhos = list(grafo.neighbors(id_central)) #Pega os vizinhos do nó
    nos_desenhos= [id_central] + vizinhos

    subgrafo = grafo.subgraph(nos_desenhos)

    labels = {}
    for no_id in subgrafo.nodes():
        labels[no_id] = grafo.nodes[no_id].get('title',no_id)
    


def main():
    arquivo = 'disney_plus_title_list.csv'
    print("O grafo esta sendo criado")
    grafo = cria_grafo(arquivo)
    print("Grafo criado com sucesso")
    




if __name__ == "__main__":
    main()