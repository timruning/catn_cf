import faiss
import os
import argparse
import json
import faiss
from faiss import normalize_L2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default="601")
parser.add_argument('--target', type=str, default="602")
args = parser.parse_args()


def to_list(vec: str):
    return [float(v) for v in vec.split(",")]


if __name__ == '__main__':
    path = f"../data/embedding/embedding_{args.source}_{args.target}"
    df_source = pd.read_csv(f"{path}/{args.source}", sep="\t", names=['id', "pid", "vector"])\
        .drop_duplicates("pid")
    df_source["vector"] = df_source["vector"].map(lambda x: to_list(x))

    df_target = pd.read_csv(f"{path}/{args.target}", sep="\t", names=['id', "pid", "vector"])\
        .drop_duplicates("pid")
    df_target["vector"] = df_target["vector"].map(lambda x: to_list(x))

    df_source_vec = np.array(list(df_source["vector"])).astype('float32')
    normalize_L2(df_source_vec)
    index = faiss.IndexFlatIP(300)
    index.train(df_source_vec)
    print(index.is_trained)

    df_target_vec = np.array(list(df_target["vector"])).astype('float32')
    normalize_L2(df_target_vec)
    index.add(df_target_vec)

    D, I = index.search(df_source_vec, 51)

    I.shape

    id_index = list(df_source["pid"])
    id_index1 = list(df_target["pid"])

    if not os.path.exists("../data/id_pair/"):
        os.mkdir("../data/id_pair/")

    file = open(f"../data/id_pair/result_{args.source}_{args.target}", "w")
    for i, v in enumerate(I):
        key = id_index[i]
        v1 = [str(id_index1[t]) for t in v]
        value = ",".join(v1)
        line = "{}\t{}\n".format(key, value)
        file.write(line)

    file = open(f"../data/id_pair/{args.source}_{args.target}", "w")
    for i, (v, w) in enumerate(zip(I, D)):
        v1 = [id_index1[t] for t in v]
        dic = {}
        dic["trigger_key"] = id_index[i]
        pairs = [{"key": int(t[0]), "score": float(t[1]), "types": "embedding#{}".format(int(t[1] * 1000) / 1000.0)} for
                 t
                 in zip(v1, w) if int(t[0]) != id_index[i]]
        dic["pairs"] = pairs
        js = json.dumps(dic)
        if i % 50000 == 49999:
            print(i, v, w)
        file.write(js)
        file.write("\n")
    file.close()

    len(set(df_source["pid"]))
    len(set(df_target["pid"]))
