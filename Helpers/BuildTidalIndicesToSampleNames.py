import os, json

filenamesToTidal={}
tidalToFilenames={}
numSamples={}

samplesFolder=os.path.join("/home", "blindelephants", "slab", "SoundSamples")
ld = os.listdir(samplesFolder)
ld.sort()

ld.pop(ld.index(".git"))

for d in ld:
    samplesFolderPath = os.path.join(samplesFolder, d)
    if os.path.isdir(samplesFolderPath):
        listSamps = os.listdir(samplesFolderPath)
        listSamps.sort()

        numSamples[d]=len(listSamps)

        for i, s in enumerate(listSamps):
            if s.split(".")[-1] in ['WAV', 'wav', 'Wav']:
                filenamesToTidal[os.path.join(d,s)]=d+":"+str(i)
                tidalToFilenames[d+":"+str(i)] = os.path.join(d,s)


import datetime

TidalToFilenamesDat = {
    'filenamesToTidal': filenamesToTidal,
    'tidalToFilenames': tidalToFilenames,
    'numSamples': numSamples,
    'generated': str(datetime.datetime.now())
}

with open(os.path.join(os.path.dirname(__file__), "TidalToFilenames.json"), "w") as f:
    json.dump(TidalToFilenamesDat, f)