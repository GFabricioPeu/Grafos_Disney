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
    
    itens =[item.strip() for item in C.split (',') if item.strip()]

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

    grafo.graph['titulo_para_id']= {}#Ajuda na hora da busca

    corpus_index = corpus.set_index('show_id')

    for show_id, row in corpus_index.iterrows():
        titulo = row.get('title', 'N/A')
        grafo.add_node(
            show_id, 
            type='Filme', # Tipo de nó
            title=row.get('title', 'N/A'), 
            cluster=row.get('cluster', -1)
        )
        if titulo != 'N/A':
            grafo.graph['titulo_para_id'][titulo.lower().strip()] = show_id
        for ator in row['cast']:
            grafo.add_node(ator, type='Ator')
            grafo.add_edge(show_id, ator, type='Tem_ator')

        for diretor in row['director']:
            grafo.add_node(diretor, type = 'Diretor')
            grafo.add_edge(show_id,diretor,type='Tem_diretor')

        for categoria in row['listed_in']:
            grafo.add_node(categoria, type='Categoria')
            grafo.add_edge(show_id, categoria, type='Tem_categoria')

        for pais in row['country']:
            grafo.add_node(pais, type='Pais')
            grafo.add_edge(show_id, pais, type='Tem_pais')


    similaridade = cosine_similarity(matriz_vetorizada)
    lim_similaridade = 0.1

    for i in range(similaridade.shape[0]):
        for j in range(i+1,similaridade.shape[1]):
            if similaridade[i,j] > lim_similaridade:
                grafo.add_edge(corpus.iloc[i]['show_id'],corpus.iloc[j]['show_id'],type = 'Similar',weight=similaridade[i,j])

    return grafo

def mostra_vizinhos(grafo,id_central):
    max_recomAda = 5

    if id_central not in grafo: #Ve se o nó realmente ta no grafo
        print("No nao esta no grafo")
        return
    
    vizinhos_cos = set()

    for vizinho in grafo.neighbors(id_central): #Pega os vizinhos do nó
        if (grafo.nodes[vizinho].get('type') == 'Filme' and 
            grafo.has_edge(id_central, vizinho) and
            grafo.edges[id_central, vizinho].get('type') == 'Similar'):
            vizinhos_cos.add(vizinho)

    #Agora pega os nós Adamic-Adar
    nos_filme = {n for n, d in grafo.nodes(data=True) if d.get('type') == 'Filme'}
    nos_Ada = nos_filme - vizinhos_cos - {id_central}
    
    bunch = [(id_central,outro_no) for outro_no in nos_Ada]
   
    Ada = nx.adamic_adar_index(grafo,bunch)

    recomAda = sorted(Ada,key = lambda item: item[2], reverse = True)
    recomTopo = recomAda[:max_recomAda]

    vizinhos_Ada = [v for u,v, score in recomTopo]

    nos_desenho = list (vizinhos_cos.union(set(vizinhos_Ada)|{id_central}))

    subgrafo = grafo.subgraph(nos_desenho).copy()

    subgrafo.clear_edges()

    for v in vizinhos_cos:
        if v in subgrafo:
            subgrafo.add_edge(id_central,v,type='Similar')

    for u, v, score in recomTopo:
        subgrafo.add_edge(u, v, weight=score, type='Adamic_Adar')

    labels = {node_id: grafo.nodes[node_id].get('title', node_id) for node_id in subgrafo.nodes()} 

    vizinhos = [node for node in subgrafo.nodes() if node != id_central]
    shells = [ [id_central], vizinhos ]
    pos = nx.shell_layout(subgrafo, shells)
    edges_cos = [(u, v) for u, v, d in subgrafo.edges(data=True) if d.get('type') == 'Similar']
    edges_aa = [(u, v) for u, v, d in subgrafo.edges(data=True) if d.get('type') == 'Adamic_Adar']

    # Parte que desenha o Grafo
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_nodes(subgrafo, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_nodes(subgrafo, pos, nodelist=[id_central], node_color='red', node_size=700)
    nx.draw_networkx_labels(subgrafo, pos, labels=labels, font_size=8)

    nx.draw_networkx_edges(subgrafo, pos, edgelist=edges_cos, edge_color='gray', style='solid', alpha=0.7, label='Similar (Descrição)')
    
    nx.draw_networkx_edges(subgrafo, pos, edgelist=edges_aa, edge_color='red', style='dashed', alpha=1.0, label='Recomendado (Adamic-Adar)')

    plt.title(f"Parecidos com: {labels[id_central]}")
    plt.axis('off')
    plt.legend()
    plt.show()



def recomenda(titulo_filme,grafo):

    print(f"\nBuscando recomendacoes parecidas com {titulo_filme}")

    mapa_de_titulos = grafo.graph.get('titulo_para_id', {})
    titulo_lower = titulo_filme.lower().strip()
    show_id = mapa_de_titulos.get(titulo_lower)

    if not show_id:
        print("Titulo nao foi encontrado")
        parece_pouco = [db_titulo for db_titulo in mapa_de_titulos.keys() if db_titulo in titulo_lower]
        
        if parece_pouco:
            parece_mais = max(parece_pouco,key=len)
            show_id = mapa_de_titulos[parece_mais]
            achou_parecido = grafo.nodes[show_id].get('title')
            print(f"\n Nao foi possivel encontrar o titulo informado. Porem foi achado um parecido: {achou_parecido}")

    if not show_id:
        print("\n Erro em achar o titulo informado no grafo")

        parecidos = [
            db_titulo for db_titulo in mapa_de_titulos.keys() 
            if titulo_lower in db_titulo
        ]
        if parecidos:

            titulos_parecidos = [grafo.nodes[mapa_de_titulos[t]].get('title') for t in parecidos]
            print(f"Você quis dizer: {titulos_parecidos[:3]}?")
        return
    
    mostra_vizinhos(grafo,show_id)





def main():
    arquivo = 'disney_plus_title_list.csv'
    print("O Grafo esta sendo criado")
    grafo = cria_grafo(arquivo)
    print("Grafo criado com sucesso")

    
    titulo_filme = input("(digite 'sair' para finalizar) Digite o nome do filme: ")
    recomenda(titulo_filme,grafo)


if __name__ == "__main__":
    main() 