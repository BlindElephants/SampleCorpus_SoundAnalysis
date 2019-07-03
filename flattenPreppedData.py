import os
import json

with open("allDat_Prepped.json", "r") as f: dat = json.load(f)
flattenedDat={
    'NAMES': [],
    'COMBINES': [],
    'COMBINES_TSNE': [],
    'STFTS': [],
    'STFTS_TSNE': [],
    'CQTS': [],
    'CQTS_TSNE': [],
    'SPECS': [],
    'SPECS_TSNE': [],
    'DURATIONS': []
}

for key, val in dat.items():
    COMBINE = []
    STFT = []
    CQT = []
    SPEC = []
    DURATION = val['duration']
    
    for k, v in val.items():
        if "stft_" in k:
            COMBINE += v
            STFT+=v
        elif "cqt_" in k:
            COMBINE += v
            CQT +=v
        elif 'spec ' in k:
            COMBINE += v
            SPEC+=v

    flattenedDat['NAMES'].append(val['sample_filename'])
    flattenedDat['COMBINES'].append(COMBINE)
    flattenedDat['STFTS'].append(STFT)
    flattenedDat['CQTS'].append(CQT)
    flattenedDat['SPECS'].append(SPEC)
    flattenedDat['DURATIONS'].append(DURATION)

with open("allDat_PreppedFlattened.json", "w") as f: json.dump(flattenedDat, f)

import numpy as np
from sklearn.manifold import TSNE

tsne_combine = TSNE(n_components=3, perplexity=60, verbose=2)
tsne_stft    = TSNE(n_components=3, perplexity=75, learning_rate=500, verbose=2)
tsne_cqt     = TSNE(n_components=3, perplexity=75, learning_rate=500, verbose=2)
tsne_spec    = TSNE(n_components=3, perplexity=60, verbose=2)

if os.path.exists("tSNE_Embeddings.json"):
    with open("tSNE_Embeddings.json", "r") as f:
        tSNE_Coords = json.load(f)
else:
    tSNE_Coords = {}

tSNE_Coords['NAMES'] = flattenedDat['NAMES']
tSNE_Coords['COMBINES_TSNE'] = tsne_combine.fit_transform(flattenedDat['COMBINES']).tolist()
tSNE_Coords['STFTS_TSNE'] = tsne_stft.fit_transform(flattenedDat['STFTS']).tolist()
tSNE_Coords['CQTS_TSNE'] = tsne_cqt.fit_transform(flattenedDat['CQTS']).tolist()
tSNE_Coords['SPECS_TSNE'] = tsne_spec.fit_transform(flattenedDat['SPECS']).tolist()

print("saving")
with open("tSNE_Embeddings.json", "w") as f: json.dump(tSNE_Coords, f)
print("done")

