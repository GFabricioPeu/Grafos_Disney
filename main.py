import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import sklearn.feature_extraction.text as skt
import sklearn.cluster as skc
from sklearn.metrics.pairwise import cosine_similarity
import re

remover = {'as','the', 'and', 'if','that','in','when','so','where','at','but','or','to'}

'''
INUTIL POREM DEU TRABALHO

def limpa_description(D):
    if not isinstance(D,str):
        return ""
    
    palavras 
    palavras = D.split()
    palavras_resultado = [palavra for palavra in palavras if palavra.lower() not in remover]

    return ' '.join(palavras_resultado)



def limpa_cast(C):
    if not isinstance(C,str):
        return []
    
    atores =[ator.strip() for ator in C.split (',') ]

    return atores


def trata_arquivo (corpus):
    
    corpus['description']= corpus['description'].apply(limpa_description)
    corpus['cast']= corpus['cast'].apply(limpa_cast)

    return corpus
'''


def cria_grafo (arquivo):
    df = pd.read_csv(arquivo)
    corpus = df[['show_id','type','title','director','cast','country','rating','listed_in','description']].fillna('')
    
   
    vectorizer = skt.TfidfVectorizer(stop_words=remover)
    matriz_vetorizada = vectorizer.fit_transform(corpus['description'])

    kmeans = skc.MiniBatchKMeans(n_clusters=100,random_state=9).fit(matriz_vetorizada)
    df['cluster'] = kmeans.labels_

    grafo = nx.Graph()

    df_index = df.set_index('show_id')
    for show_id, row in df_index.iterrows():
        grafo.add_node(
            show_id, 
            title=row.get('title', 'N/A'), 
            type=row.get('type', 'N/A'),
            cluster=row.get('cluster', -1)
        )

    similaridade = cosine_similarity(matriz_vetorizada)
    lim_similaridade = 0.1
    for i in range(similaridade.shape[0]):
        for j in range(i+1,similaridade.shape[1]):
            if similaridade[i,j] > lim_similaridade:
                grafo.add_edge()

def main():
    arquivo = 'disney_plus_title_list.csv'
    grafo = cria_grafo(arquivo)



if __name__ == "__main__":
    main()