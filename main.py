import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import sklearn.feature_extraction.text as skt
import sklearn.cluster as skc

remover = {'as','the', 'and', 'if','that','in','when','so','where','at','but','or','to'}


def limpa_description(D):
    if not isinstance(D,str):
        return ""
    
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



def cria_grafo (arquivo):
    df = pd.read_csv(arquivo)
    corpus = df['show_id','title','director','cast','country','listed_in','description'].fillna('').tolist()
    corpus = trata_arquivo(corpus)

    vectorizer = skt.TfidfVectorizer()
    matriz_vetorizada = vectorizer.fit_transform(corpus)
    kmeans = skc.MiniBatchKMeans(n_clusters=100,random_state=9).fit(matriz_vetorizada)

    grafo = nx.Graph()




def main():
    arquivo = 'disney_plus_title_list.csv'
    grafo = cria_grafo(arquivo)



if __name__ == "__main__":
    main()