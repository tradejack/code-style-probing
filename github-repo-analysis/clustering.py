# assumes file system to walk
# does relative casing counts per file

import os
from ast_gloss import ast_parse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

directory = "data"

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def case_clustering(path):
    i = 0
    data = []
    kmeans = KMeans(n_clusters= 4)
    pca = PCA(2)
    for folder in get_immediate_subdirectories(path):
        scripts_processed = 0
        sub_path = "data/"+folder
        for root, directory, files in os.walk(sub_path):
            if scripts_processed > 25:
                    break
            for file_name in files:
                if file_name.endswith(".py"):
                    file_path = os.path.join(root, file_name)
                    with(open(file_path, 'r', encoding='latin-1')) as fp:
                        code = fp.read()
                        fp.close()
                    try:
                        parse_dict = ast_parse(code)
                        scripts_processed += 1
                    except:
                        print(f"Couldn't parse file {file_name}, skipping")
                    total_vars = parse_dict['var_total']
                    if total_vars == 0:
                        continue
                    sc = parse_dict['snake_case'] / total_vars
                    lc = parse_dict['lower_camel'] / total_vars
                    uc = parse_dict['upper_camel'] / total_vars
                    lo = parse_dict['lower'] / total_vars
                    data += [np.array([sc, lc, uc, lo])]
    data_array = np.stack(data, axis = 0)
    df = pca.fit_transform(data_array)
    label = kmeans.fit_predict(df)
    centroids = kmeans.cluster_centers_
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.show()

case_clustering(directory)