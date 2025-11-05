import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import sklearn.feature_extraction.text as skt
import sklearn.cluster as skc
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
    corpus = df[['show_id','type','title','director','cast','country','rating','listed_in','description']].fillna('').tolist()
    
    for collums in ['type','title','director','cast','country','listed_in','description']:

        vectorizer = skt.TfidfVectorizer(stop_words=remover)
        matriz_vetorizada = vectorizer.fit_transform(corpus)


        
    kmeans = skc.MiniBatchKMeans(n_clusters=100,random_state=9).fit(matriz_vetorizada)

    grafo = nx.Graph()




def main():
    arquivo = 'disney_plus_title_list.csv'
    grafo = cria_grafo(arquivo)



if __name__ == "__main__":
    main()