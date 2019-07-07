import os, json

class FindSimilarSamples():
    def __init__(self, ClusteringSourceSize=500):
        self.ClusteringSourceSize=ClusteringSourceSize
        assert self.ClusteringSourceSize in [2, 3, 4, 8, 12, 16, 20, 24, 30, 40, 60, 80, 120, 200, 500]
        print("Checking if TidalToFilenames.json exists.")
        TidalToFilenamesFilePath=os.path.join(os.path.dirname(__file__), "..", "Helpers", "TidalToFilenames.json")
        assert os.path.exists(TidalToFilenamesFilePath)
        
        print("Loading Tidal names to filenames conversion tables.")
        with open(TidalToFilenamesFilePath, "r") as f: TidalToFilenamesDat=json.load(f)
        self.TidalToFilenames=TidalToFilenamesDat['tidalToFilenames']
        self.FilenamesToTidal=TidalToFilenamesDat['filenamesToTidal']
        self.NumSamples=TidalToFilenamesDat['numSamples']
        print("[DONE]")

        print("Loading Cluster Data.")

        clusteringDatPath=os.path.join(os.path.dirname(__file__), "..", "Clustering", "JSON")

        with open(os.path.join(clusteringDatPath, "COMBINES_TSNE_"+str(ClusteringSourceSize)+".json"), "r") as f:
            self.combines_dat=json.load(f)
        with open(os.path.join(clusteringDatPath, "CQTS_TSNE_"+str(ClusteringSourceSize)+".json"), "r") as f:
            self.cqts_dat=json.load(f)
        with open(os.path.join(clusteringDatPath, "SPECS_TSNE_"+str(ClusteringSourceSize)+".json"), "r") as f:
            self.specs_dat=json.load(f)
        with open(os.path.join(clusteringDatPath, "STFTS_TSNE_"+str(ClusteringSourceSize)+".json"), "r") as f:
            self.stfts_dat=json.load(f)

        print("[DONE]")
        self.counts={}
        self.likeness=[[],[],[],[]]

    def addIncrToCounts(self, in_clusters):
        for x in in_clusters:
            if x in self.counts:
                self.counts[x]+=1
            else:
                self.counts[x]=1

    def getSimilarSamples(self, sampleName, likeness=3):
        if likeness > 3: likeness=3
        if likeness < 0: likeness=0

        self.counts={}
        self.likeness=[[], [], [], []]

        try:
            splitSampleName=sampleName.split(":")
            splitSampleName[1]=str(int(splitSampleName[1])%int(self.NumSamples[splitSampleName[0]]))
            fn=self.TidalToFilenames[":".join(splitSampleName)]
            idx=self.combines_dat['NAMES'].index(fn)
            
            combines_cluster=self.combines_dat['out_predict'][idx]
            cqts_cluster=self.cqts_dat['out_predict'][idx]
            specs_cluster=self.specs_dat['out_predict'][idx]
            stfts_cluster=self.stfts_dat['out_predict'][idx]

            in_cluster_combines = [i for i,x in enumerate(self.combines_dat['out_predict']) if x==combines_cluster]
            in_cluster_cqts = [i for i,x in enumerate(self.cqts_dat['out_predict']) if x==cqts_cluster]
            in_cluster_specs= [i for i,x in enumerate(self.specs_dat['out_predict']) if x==specs_cluster]
            in_cluster_stfts= [i for i,x in enumerate(self.stfts_dat['out_predict']) if x==stfts_cluster]

            self.addIncrToCounts(in_cluster_combines)
            self.addIncrToCounts(in_cluster_cqts)
            self.addIncrToCounts(in_cluster_specs)
            self.addIncrToCounts(in_cluster_stfts)

            for k,v in self.counts.items(): self.likeness[v-1].append(self.FilenamesToTidal[self.combines_dat['NAMES'][k]])
        except IndexError as e:
            pass
        except KeyError as e:
            pass
        return self.likeness[likeness]
