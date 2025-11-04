import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as ntx
import sklearn.feature_extraction.text as skt
import sklearn.cluster as skc

 
df = pd.read_csv('disney_plus_title_list.csv')
corpus = df['description'].tolist()
vectorizer = skt.TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

#print(X.shape[1])

kmeans = skc.MiniBatchKMeans(n_clusters=100).fit(X)
#print(kmeans.cluster_centers_)


