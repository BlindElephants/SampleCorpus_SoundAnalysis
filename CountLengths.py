import json
import os

durations = []
numberDecimalsToRoundTo=1
useLogYAxis=True

import numpy as np
import matplotlib.pyplot as plt

durationsJsonFile = "durations.json"

if os.path.exists(durationsJsonFile):
    with open (durationsJsonFile, "r") as f:
        durations = json.load(f)
else:
    samplesData = os.path.join(os.path.dirname(__file__), "SampleAnalysisData", "Samples")

    ld = os.listdir(samplesData)
    ld.sort()
    print("Samples found: {}".format(len(ld)))

    for x in range(len(ld)):
        print(x)
        with open(os.path.join(samplesData, ld[x]), "r") as f:
            d = json.load(f)
        durations.append(d['duration'])
    durations.sort()

    with open (durationsJsonFile, "w") as f:
        json.dump(durations, f)

for x in range(len(durations)):
    durations[x] = np.around(durations[x], decimals=numberDecimalsToRoundTo)
u, c = np.unique(durations, return_counts=True)

plt.figure(figsize=(19.2,10.8))
plt.title("TidalCycles Samples Collection Durations ({} samples)".format(len(durations)))
plt.bar(u, c, width=0.2, log=useLogYAxis)
plt.xticks(np.arange(0,80,5))
plt.xlabel("Sample Length (seconds)")
plt.ylabel("Number Samples")
plt.savefig("TidalCyclesSamplesDurationsBarPlot.png")

