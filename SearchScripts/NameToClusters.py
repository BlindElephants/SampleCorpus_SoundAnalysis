import os
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("SampleName", type=str)
parser.add_argument("--ClusteringSourceSize", type=int, default=500,  help="[2, 3, 4, 8, 12, 16, 20, 24, 30, 40, 60, 80, 120, 200, 500]")
args=parser.parse_args()

assert args.ClusteringSourceSize in [2, 3, 4, 8, 12, 16, 20, 24, 30, 40, 60, 80, 120, 200, 500]

with open(os.path.join(os.path.dirname(__file__), "..", "Clustering", "JSON", "COMBINES_TSNE_"+str(args.ClusteringSourceSize)+".json"), "r") as f: combines_dat = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "..", "Clustering", "JSON", "CQTS_TSNE_"+str(args.ClusteringSourceSize)+".json"), "r") as f: cqts_dat = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "..", "Clustering", "JSON", "SPECS_TSNE_"+str(args.ClusteringSourceSize)+".json"), "r") as f: specs_dat = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "..", "Clustering", "JSON", "STFTS_TSNE_"+str(args.ClusteringSourceSize)+".json"), "r") as f: stfts_dat = json.load(f)

idx_combines = combines_dat['NAMES'].index(args.SampleName)
idx_cqts = cqts_dat['NAMES'].index(args.SampleName)
idx_specs= specs_dat['NAMES'].index(args.SampleName)
idx_stft = stfts_dat['NAMES'].index(args.SampleName)
print("Name Indices:\t{}\t{}\t{}\t{}".format(idx_combines, idx_cqts, idx_specs, idx_stft))

combines_cluster = combines_dat['out_predict'][idx_combines]
cqts_cluster = cqts_dat['out_predict'][idx_cqts]
specs_cluster= specs_dat['out_predict'][idx_specs]
stfts_cluster= stfts_dat['out_predict'][idx_stft]

print("Cluster ids:\t{}\t{}\t{}\t{}".format(combines_cluster, cqts_cluster, specs_cluster, stfts_cluster))

in_cluster_combines = [i for i,x in enumerate(combines_dat['out_predict']) if x==combines_cluster]
in_cluster_cqts = [i for i,x in enumerate(cqts_dat['out_predict']) if x==cqts_cluster]
in_cluster_specs= [i for i,x in enumerate(specs_dat['out_predict']) if x==specs_cluster]
in_cluster_stfts= [i for i,x in enumerate(stfts_dat['out_predict']) if x==stfts_cluster]

counts = {}

for x in in_cluster_combines:
    if x in counts:
        counts[x] += 1
    else:
        counts[x] = 1

for x in in_cluster_cqts:
    if x in counts:
        counts[x] += 1
    else:
        counts[x]=1

for x in in_cluster_specs:
    if x in counts:
        counts[x] += 1
    else:
        counts[x]=1

for x in in_cluster_stfts:
    if x in counts:
        counts[x]+=1
    else:
        counts[x]=1

likeness=[[],[],[],[]]

for k,v in counts.items():
    likeness[v-1].append(combines_dat['NAMES'][k])

print("1 alike: {}".format(likeness[0]))
print("2 alike: {}".format(likeness[1]))
print("3 alike: {}".format(likeness[2]))
print("4 alike: {}".format(likeness[3]))
