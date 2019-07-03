import torch
import torch.nn as nn

class LatentPredictorFromAnalysis(nn.Module):
    # def __init__(self, input_size, hidden_size=128,tsne_dimensions=3, n_clusters_out=10):
    def __init__(self, input_size, hidden_size=128, tsne_dimensions=3):
        super(LatentPredictorFromAnalysis, self).__init__()
        self.input_size = input_size
        self.hidden_size= hidden_size
        self.tsne_dimensions = tsne_dimensions
        # self.n_clusters_out = n_clusters_out

        self.fc = nn.Linear(self.input_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.relu = nn.ReLU()

        self.fc2_1= nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_2= nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_3= nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_4= nn.Linear(self.hidden_size, self.tsne_dimensions)

        # self.fc3_1= nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc3_2= nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc3_3= nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc3_4= nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc3_5= nn.Linear(self.hidden_size, self.n_clusters_out)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, inp, bsz=1):
        out = self.fc(inp.view(bsz, self.input_size))
        out = self.relu(out)
        out = self.relu(self.fc1(out))

        out2 = self.relu(self.fc2_1(out))
        out2 = self.relu(self.fc2_2(out2))
        out2 = self.relu(self.fc2_3(out2))
        out2 = self.fc2_4(out2)

        # out3 = self.relu(self.fc3_1(out))
        # out3 = self.relu(self.fc3_2(out3))
        # out3 = self.relu(self.fc3_3(out3))
        # out3 = self.relu(self.fc3_4(out3))
        # out3 = self.softmax(self.fc3_5(out3))

        # return out2, out3
        return out2