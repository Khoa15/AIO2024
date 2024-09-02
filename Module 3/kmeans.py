from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="KMeans")

parser.add_argument("--n_clusters", type=int, default=3)
parser.add_argument("--columns", default="0,1", help="Index of columns")
args = parser.parse_args()
print(args)
if args.n_clusters != None:
    data = load_iris().data
    columns = list(map(int, args.columns.split(",")))
    samples = data[:, [col for col in columns]]

    
    kmeans = KMeans(args.n_clusters, n_init="auto", random_state=0)
    kmeans.fit(samples)
    print(f'Score: {kmeans.score(samples)}')
    print(f'Cluster centers: {kmeans.cluster_centers_}')