import os
import json
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb 

with open("tSNE_Embeddings.json", "r") as f:
    dat = json.load(f)

# n_clusters = [2, 3, 4, 8, 12, 16, 20, 24, 30, 40, 60, 80, 120, 200, 500]
n_clusters = [500]

for key, val in dat.items():
    if key != "NAMES":
        v = np.array(val)

        for nc in n_clusters:
            print("performing clustering algorithm: {}".format(key))
            k=KMeans(n_clusters=nc, verbose=0)
            k.fit(v)
            out_predict = k.predict(v)
            sv = np.ones((out_predict.shape[0],2))
            hsv = np.concatenate((out_predict.reshape(-1, 1)/nc, sv), axis=1)

            out_transform=k.transform(v)

            thisClusterDat = {
                'NAMES': dat['NAMES'],
                'source': key,
                'n_clusters': nc,
                'out_predict': out_predict.tolist(),
                'out_transform': out_transform.tolist()
            }

            filename = key+"_"+str(nc)

            with open(os.path.join(os.path.dirname(__file__), "Clustering", "JSON", filename+".json"), "w") as f:
                json.dump(thisClusterDat, f)

            plt.figure(figsize=(19.2,10.8))
            plt.subplot(1, 2, 1)
            plt.title(filename +  " x,y (color=cluster classification)")
            plt.scatter(v[:,0], v[:,1], s=None, cmap=plt.get_cmap("jet"), c=hsv_to_rgb(hsv), alpha=0.4)

            plt.subplot(1, 2, 2)
            plt.title(filename + " z,y (color=cluster classification)")
            plt.scatter(v[:,2], v[:,1], s=None, cmap=plt.get_cmap("jet"), c=hsv_to_rgb(hsv), alpha=0.4)

            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__), "Clustering", "Figures",filename+"_vis.png"))
            plt.close()

