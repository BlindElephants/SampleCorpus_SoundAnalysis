import os, json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ClusteringSourceSize", type=int, default=500)
parser.add_argument("--SendToAddr", type=str, default="localhost")
parser.add_argument("--SendToPort", type=int, default=8881)
args=parser.parse_args()

assert args.ClusteringSourceSize in [2, 3, 4, 8, 12, 16, 20, 24, 30, 40, 60, 80, 120, 200, 500]

print("Checking if TidalToFilenames.json exists.")
tidalToFilenamesFilePath = os.path.join(os.path.dirname(__file__), "..", "Helpers", "TidalToFilenames.json")

if not os.path.exists(tidalToFilenamesFilePath):
    print("Does not exist. Generating now.")
    os.system("python ../Helpers/BuildTidalIndicesToSampleNames.py")

print("Loading Tidal names to filenames conversion tables.")
with open(tidalToFilenamesFilePath, "r") as f: TidalToFilenamesDat = json.load(f)
tidalToFilenames = TidalToFilenamesDat['tidalToFilenames']
filenamesToTidal = TidalToFilenamesDat['filenamesToTidal']
numSamples = TidalToFilenamesDat["numSamples"]
print("[DONE]")

print("Loading Cluster Data.")

with open(os.path.join(os.path.dirname(__file__), "..", "Clustering", "JSON", "COMBINES_TSNE_"+str(args.ClusteringSourceSize)+".json"), "r") as f: combines_dat = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "..", "Clustering", "JSON", "CQTS_TSNE_"+str(args.ClusteringSourceSize)+".json"), "r") as f: cqts_dat = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "..", "Clustering", "JSON", "SPECS_TSNE_"+str(args.ClusteringSourceSize)+".json"), "r") as f: specs_dat = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "..", "Clustering", "JSON", "STFTS_TSNE_"+str(args.ClusteringSourceSize)+".json"), "r") as f: stfts_dat = json.load(f)

print("[DONE]")

from pythonosc import udp_client, dispatcher, osc_server

client = udp_client.SimpleUDPClient(args.SendToAddr, args.SendToPort)

counts = {}
likeness = [[],[],[],[]]

def getSimilarSamples(unused_addr, *args):
    global combines_dat, cqts_dat, specs_dat, stfts_dat
    global counts, likeness
    global client
    global numSamples
    try:
        splitSampleName = args[0].split(":")
        splitSampleName[1]=str(int(splitSampleName[1])%int(numSamples[splitSampleName[0]]))
        fn = tidalToFilenames[":".join(splitSampleName)]
        idx = combines_dat['NAMES'].index(fn)
        combines_cluster = combines_dat['out_predict'][idx]
        cqts_cluster = cqts_dat['out_predict'][idx]
        specs_cluster= specs_dat['out_predict'][idx]
        stfts_cluster= stfts_dat['out_predict'][idx]

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
            likeness[v-1].append(filenamesToTidal[combines_dat['NAMES'][k]])

        client.send_message("/most_alike", likeness[3])
    except ValueError as e:
        print("ValueError:\t{}".format(e))
        client.send_message("/error", "sample name invalid")
    
def getLastLikeness(unused_addr, likeness_level):
    global likeness, client
    if likeness_level in [1, 2, 3, 4]:
        client.send_message('/likeness', likeness[likeness_level-1])

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/findSimilarSamples", getSimilarSamples)
dispatcher.map('/getLastLikeness', getLastLikeness)
dispatcher.map("/test", print)

import socket
hostname = socket.gethostname()
myIpAddr = socket.gethostbyname(hostname)
print("Server hostname: {}\t\tServer IP Address: {}".format(hostname, myIpAddr))

print("Starting server.")
server = osc_server.ThreadingOSCUDPServer((myIpAddr, 8889), dispatcher)
server.serve_forever()
