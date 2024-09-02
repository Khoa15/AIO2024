from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="KNeighborsClassifier")

parser.add_argument("--n_neighbors", type=int, default=3)
parser.add_argument("--data", default="./dataset/diabetes.csv", help="The filename dataset")
parser.add_argument("--columns", default="0,1,2,3,4,5,6,7,8", help="Index of columns")
parser.add_argument("--target", default=8, help="Should be in columns")
args = parser.parse_args()
print(args)

if args.data != None and args.n_neighbors != None:
    df = pd.read_csv(args.data).values
    columns = list(map(int, args.columns.split(",")))
    columns.remove(args.target)
    X = df[:, [col for col in columns]]
    y = df[:, args.target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.3, random_state=42)
    
    knn = KNeighborsClassifier(args.n_neighbors)
    knn.fit(X_train, y_train)
    print(f'Score: {knn.score(X_test, y_test)}')