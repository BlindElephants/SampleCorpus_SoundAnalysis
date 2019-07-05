import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1e-8)
parser.add_argument("--n_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=500)
args = parser.parse_args()

hidden_size=args.hidden_size
lr=args.lr

preppedFlattenedDataFilename= os.path.join(os.path.dirname(__file__), "..", "allDat_PreppedFlattened.json")
with open(preppedFlattenedDataFilename, "r") as f: combinedSourceData = json.load(f)['COMBINES']
print(len(combinedSourceData))

tsneEmbeddingsDataFilename = os.path.join(os.path.dirname(__file__), "..", "tSNE_Embeddings.json")
with open(tsneEmbeddingsDataFilename, "r") as f: tsneEmbeddings = json.load(f)['COMBINES_TSNE']
print(len(tsneEmbeddings))

assert len(combinedSourceData)==len(tsneEmbeddings)

combined = list(zip(combinedSourceData, tsneEmbeddings))
from random import shuffle
shuffle(combined)
combinedSourceData[:], tsneEmbeddings[:] = zip(*combined)

import torch
import torch.nn as nn
from torch import optim
import Latent_Predictor_FromAnalysis as Model

m = Model.LatentPredictorFromAnalysis(len(combinedSourceData[0]), hidden_size=hidden_size, tsne_dimensions=len(tsneEmbeddings[0])).cuda()
print(m)

optimizer = optim.Adam(m.parameters(), lr=lr, weight_decay=args.weight_decay)
tsneCrit = nn.L1Loss()
n_epochs = args.n_epochs
batchsize=args.batch_size

numfullbatches = len(combinedSourceData)//batchsize
remainder = len(combinedSourceData)%batchsize
print("Number full batches: {}".format(numfullbatches))
print("Remainder: {}".format(remainder))

valStartIdx = (numfullbatches-1)*batchsize
validationInp =  combinedSourceData[valStartIdx:]
validationTsne= tsneEmbeddings[valStartIdx:]

allTrainingLosses  =[]
allValidationLosses=[]

for epoch in range(n_epochs):
    epochLoss = 0.0
    m.train()
    for b in range(numfullbatches-1):
        inp = combinedSourceData[b*batchsize:(b+1)*batchsize]
        tsneTar = tsneEmbeddings[b*batchsize:(b+1)*batchsize]
        inp = torch.tensor(inp, dtype=torch.float).cuda()
        tsneTar=torch.tensor(tsneTar, dtype=torch.float).cuda()
        tsneOut = m(inp, batchsize)
        optimizer.zero_grad()
        loss = tsneCrit(tsneOut, tsneTar)
        loss.backward()
        optimizer.step()

        epochLoss+=loss.item()
    print("Epoch {} of {} complete. Loss: {}".format(epoch+1, n_epochs, epochLoss/((numfullbatches-1)*batchsize)), end='')
    allTrainingLosses.append(epochLoss/((numfullbatches-1)*batchsize))
    
    valInp = torch.tensor(validationInp, dtype=torch.float).cuda()
    valTsne= torch.tensor(validationTsne, dtype=torch.float).cuda()

    m.eval()
    tsneOut = m(valInp, valInp.shape[0])
    loss = tsneCrit(tsneOut, valTsne)
    print("\t\tValidation Loss: {}".format(loss.item()/valInp.shape[0]))
    allValidationLosses.append(loss.item()/valInp.shape[0])

torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), "Trained_Models", "Final_Model.pt"))
torch.save(optimizer.state_dict(), os.path.join(os.path.dirname(__file__), "Trained_Models", "Final_Optimizer.pt"))

import matplotlib.pyplot as plt
plt.figure(figsize=(19.2,10.8))
plt.title("Combined Spectral/STFT/CQT Analysis -> TSNE coords & Cluster Ids.\nHidden Size={}\tLR={}\nbatchsize={}\tnum_epochs={}".format(hidden_size, lr, batchsize, n_epochs))
plt.plot(allTrainingLosses, label="training losses")
plt.plot(allValidationLosses, label="validation losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 0.02)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "Figures", "trainingLosses.png"))