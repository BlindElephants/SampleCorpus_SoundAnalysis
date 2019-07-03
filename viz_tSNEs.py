import os
import json
import matplotlib.pyplot as plt
import numpy as np

with open("tSNE_Embeddings.json", "r") as f:
    dat = json.load(f)

for key, val in dat.items():
    if key != "NAMES":
        plt.figure(figsize=(19.2,10.8))
        v = np.array(val)
        plt.subplot(1,2,1)
        plt.title(key+" x,y (color=z)")
        plt.scatter(v[:,0], v[:,1], c=v[:,2], alpha=0.25)
        plt.subplot(1,2,2)
        plt.title(key+" z,y (color=x)")
        plt.scatter(v[:,2], v[:,1], c=v[:,0], alpha=0.25)

        plt.tight_layout()
        plt.savefig(key+"_vis.png")
        plt.close()
