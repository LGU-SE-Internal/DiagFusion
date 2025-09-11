import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F



class RGCNClassifier(nn.Module):
    """
    两层RGCN+最大池化+线性分类器
    """

    def __init__(self, in_dim, hidden_dim, n_classes, etype):
        super(RGCNClassifier, self).__init__()
        self.etype = etype
        n_rels = len(set([e.item() for e in etype]))
        self.conv1 = dglnn.RelGraphConv(in_dim, hidden_dim, n_rels)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, n_rels)
        self.pool = dglnn.MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        etype = self.etype.repeat((g.num_edges() // len(self.etype)))
        h = F.relu(self.conv1(g, h, etype))
        h = F.relu(self.conv2(g, h, etype))
        h = self.pool(g, h)
        return self.classify(h)

    def get_embeds(self, g, h, pool=False):
        etype = self.etype.repeat((g.num_edges() // len(self.etype)))
        h = F.relu(self.conv1(g, h, etype))
        h = F.relu(self.conv2(g, h, etype))
        if pool:
            h = self.pool(g, h)
        return h



class TAGClassifier(nn.Module):
    """
    两层TAGConv+最大池化+线性分类器
    """

    def __init__(self, in_dim, hidden_dim, n_classes):
        super(TAGClassifier, self).__init__()
        self.conv1 = dglnn.TAGConv(in_dim, hidden_dim, activation=F.relu)
        self.conv2 = dglnn.TAGConv(hidden_dim, hidden_dim, activation=F.relu)
        self.pool = dglnn.MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        h = self.pool(g, h)
        return self.classify(h)

    def get_embeds(self, g, h, pool=False):
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        if pool:
            h = self.pool(g, h)
        return h
