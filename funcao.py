import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import re

remover = {'as','the', 'and', 'if','that','in','when','so','where','at','but','or','to'}

def limpa_description(D):#Limpa descrição e titulos
    if not isinstance(D,str):
        return ""

    palavras = D.split()
    palavras_resultado = [palavra for palavra in palavras if palavra.lower() not in remover]

    return ' '.join(palavras_resultado)

def limpa_lista(C): #limpa cada coluna do arquivo
    if not isinstance(C,str):
        return []
    
    itens =[item.strip() for item in C.split (',') if item.strip()]

    return itens


def trata_arquivo (corpus): #Limpa o Arquivo
    
    corpus['description']= corpus['description'].apply(limpa_description)
    colunas = ['cast','director','listed_in','country',]
    for col in colunas:
        corpus[col] = corpus[col].apply(limpa_lista)
    
    return corpus



def cria_grafo (arquivo):

    df = pd.read_csv(arquivo) #Abre o arquivo
    corpus = df[['show_id','type','title','director','cast','country','rating','listed_in','description']].fillna('')
    
    corpus = trata_arquivo(corpus) # Trata o arquivo
   
    corpus['titulo_limpo'] = corpus['title'].apply(limpa_description)#"limpa" os titulos

    # Feito para levar em conta o titulo tambem 
    # Caso queira deixar a matriz vetorizada ['combinado'] como comentario e tirar o da ['description']
    corpus['combinado'] = (corpus['titulo_limpo'] +' ')*2 + corpus['description'] #combina descrição e titulos

    vectorizer = TfidfVectorizer()
    matriz_vetorizada = vectorizer.fit_transform(corpus['combinado'])
    # matriz_vetorizada = vectorizer.fit_transform(corpus['description'])


    kmeans = MiniBatchKMeans(n_clusters=100,random_state=9, n_init=10).fit(matriz_vetorizada) # cria clusters
    corpus['cluster'] = kmeans.labels_

    grafo = nx.Graph() #inicia o grafo

    grafo.graph['titulo_para_id']= {}#Ajuda na hora da busca

    corpus_index = corpus.set_index('show_id') #usar o id como index

    for show_id, row in corpus_index.iterrows(): # cria os nós
        titulo = row.get('title', 'N/A')
        grafo.add_node(
            show_id, 
            type='Titulo', # Tipo de nó
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


    similaridade = cosine_similarity(matriz_vetorizada) #Similaridade de cosseno
    lim_similaridade = 0.3

    for i in range(similaridade.shape[0]): #cria as vertices
        for j in range(i+1,similaridade.shape[1]):
            if similaridade[i,j] > lim_similaridade:
                grafo.add_edge(corpus.iloc[i]['show_id'],corpus.iloc[j]['show_id'],type = 'Similar',weight=similaridade[i,j])

    return grafo



def mostra_vizinhos(grafo,id_central):
    max_recomAda = 5

    if id_central not in grafo: #Ve se o nó realmente ta no grafo
        print("No nao esta no grafo")
        return
    
    #Vizinhos por texto
    vizinhos_cos = set()

    for vizinho in grafo.neighbors(id_central): #Pega os vizinhos do nó
        if (grafo.nodes[vizinho].get('type') == 'Titulo' and 
            grafo.has_edge(id_central, vizinho) and
            grafo.edges[id_central, vizinho].get('type') == 'Similar'):
            vizinhos_cos.add(vizinho)

    #Começa o calculo do Adamic Adar geral

    #Pega todos os titulos do Grafo
    nos_titulos = {n for n, d in grafo.nodes(data=True) if d.get('type') == 'Titulo'}

    nos_candidatos = nos_titulos - {id_central}
    
    bunch = [(id_central,outro_no) for outro_no in nos_candidatos]
    Adamic_Adar = nx.adamic_adar_index(grafo,bunch)

    #Pega o top 5
    recomAda = sorted(Adamic_Adar,key = lambda item: item[2], reverse = True)
    recomTopo = recomAda[:max_recomAda] 

    print(f"\n--- Top {max_recomAda} Recomendações (Adamic-Adar) ---")
    
    print(f"\n--- Top {max_recomAda} Conexões Estruturais (Adamic-Adar) ---")
    top_aa_ids = []
    for u, v, score in recomTopo:
        titulo = grafo.nodes[v].get('title', 'N/A')
        extra = " (Também é similar por texto)" if v in vizinhos_cos else " (Novo por Adamic Adar)"
        print(f"Filme: {titulo} | Score: {score:.4f}{extra}")
        top_aa_ids.append(v)
    
    print("-----------------------------------------------------")

    #Uniao dos vizinhos de texto + Top 5 Adamic Adar + No central
    nos_desenho = list(vizinhos_cos.union(set(top_aa_ids)) | {id_central})

    subgrafo = grafo.subgraph(nos_desenho).copy()
    subgrafo.clear_edges()

    #Adiciona as arestas

    for v in vizinhos_cos:
        if v in subgrafo:
            subgrafo.add_edge(id_central,v,type='Similar')

    for u, v, score in recomTopo:
        subgrafo.add_edge(id_central, v, weight=score, type='Adamic_Adar')

    labels = {node_id: grafo.nodes[node_id].get('title', node_id) for node_id in subgrafo.nodes()} 
    '''
    SHELLS_LAYOUT
    vizinhos = [node for node in subgrafo.nodes() if node != id_central]
    shells = [ [id_central], vizinhos ]
    pos = nx.shell_layout(subgrafo, shells)
    '''
    pos = nx.spring_layout(subgrafo,seed = 42, k=0.8)
    
    edges_cos = []
    edges_aa = []

    for u, v, d in subgrafo.edges(data=True):
        if d.get('type') == 'Adamic_Adar':
            edges_aa.append((u,v))
        elif d.get('type') == 'Similar':
            edges_cos.append((u,v))
    
    color_map = []
    for node in subgrafo.nodes():
        if node == id_central:
            color_map.append('red')
        elif node in top_aa_ids:
            color_map.append('lightgreen') 
        else:
            color_map.append('lightblue')

    edges_cos = [(u, v) for u, v, d in subgrafo.edges(data=True) if d.get('type') == 'Similar']
    edges_aa = [(u, v) for u, v, d in subgrafo.edges(data=True) if d.get('type') == 'Adamic_Adar']

    # Parte que desenha o Grafo

    #Tamanho da figura
    plt.figure(figsize=(15, 15))

    #Faz os nos e labels
    nx.draw_networkx_nodes(subgrafo, pos, node_color=color_map, node_size=600)
    nx.draw_networkx_labels(subgrafo, pos, labels=labels, font_size=8)

    #Arestas nos de similaridade de texto
    nx.draw_networkx_edges(subgrafo, pos, edgelist=edges_cos, edge_color='gray', style='solid', alpha=0.6, label='Similaridade Texto')

    #Arestas nos Top 5 Adamic Adar
    nx.draw_networkx_edges(subgrafo, pos, edgelist=edges_aa, edge_color='red', style='dashed', width=2.5, label='Top 5 Adamic-Adar')

    plt.title(f"Parecidos com: {labels[id_central]}")
    plt.axis('off')
    plt.legend()
    plt.show()


def recomenda(titulo_usuario,grafo):

    print(f"\nBuscando recomendacoes parecidas com {titulo_usuario}")

    mapa_de_titulos = grafo.graph.get('titulo_para_id', {}) #Acha o nó no set
    titulo_lower = titulo_usuario.lower().strip() #Nó pelado minusculo 
    show_id = mapa_de_titulos.get(titulo_lower)

    if not show_id: #se não achou direto do que o usuario escreveu
        print("Titulo nao foi encontrado")
        parece_pouco = [db_titulo for db_titulo in mapa_de_titulos.keys() if db_titulo in titulo_lower]
        
        if parece_pouco: #acha o que mais parece com o que foi escrito
            parece_mais = max(parece_pouco,key=len)
            show_id = mapa_de_titulos[parece_mais]
            achou_parecido = grafo.nodes[show_id].get('title')
            print(f"\n Nao foi possivel encontrar o titulo informado. Porem foi achado um parecido: {achou_parecido}")

    if not show_id:#se nada parece com o que o burro do usuario escreveu
        print("\n Erro em achar o titulo informado no grafo")

        parecidos = [
            db_titulo for db_titulo in mapa_de_titulos.keys() 
            if titulo_lower in db_titulo
        ]
        if parecidos:

            titulos_parecidos = [grafo.nodes[mapa_de_titulos[t]].get('title') for t in parecidos]
            print(f"Você quis dizer: {titulos_parecidos[:3]}?") #da uma recomendação pro burro
        return
    
    mostra_vizinhos(grafo,show_id)