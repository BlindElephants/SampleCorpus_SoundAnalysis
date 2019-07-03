import os
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ClusteringSource", type=str, default="SPECS", help="[COMBINES, CQTS, SPECS, STFTS]")
parser.add_argument("--ClusteringSourceSize", type=int, default=500, help="[2, 3, 4, 8, 12, 16, 20, 24, 30, 40, 60, 80, 120, 200, 500]")
parser.add_argument("--PrintCluster", type=int, default=0)
args = parser.parse_args()

assert args.ClusteringSource in ["COMBINES", "CQTS", "SPECS", "STFTS"], "[ERROR] Invalid ClusteringSource: {}. Must be one of [COMBINES, CQTS, SPECS, STFTS]."
assert args.ClusteringSourceSize in [2, 3, 4, 8, 12, 16, 20, 24, 30, 40, 60, 80, 120, 200, 500]
assert args.PrintCluster>=0 and args.PrintCluster<args.ClusteringSourceSize

with open(os.path.join(os.path.dirname(__file__), "..", "Clustering", "JSON", args.ClusteringSource+"_TSNE_"+str(args.ClusteringSourceSize)+".json"), "r") as f:clusterIds=json.load(f)
found = [i for i,x in enumerate(clusterIds['out_predict']) if x==args.PrintCluster]

print("\nFound {} samples.\n".format(len(found)))
for f in found: print(clusterIds['NAMES'][f])
print("")
