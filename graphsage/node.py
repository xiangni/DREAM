import numpy as np

class Operator(object):
  def __init__(self, idx, cpu, payload, ins, outs):
    self.idx = idx
    self.cpu = cpu
    self.payload = payload
    self.ins = ins
    self.outs = outs
    self.neighbors = ins + outs
    self._build_features()

  def _build_features(self):
    self.feats = np.array([self.cpu, self.payload], dtype=np.float32)

class Graph(object):
  def __init__(self, idx, num_nodes):
    self.g_id = idx
    self.num_nodes = num_nodes
    self.node = []
  
  def add_node(self, one_node):
    self.node.append(one_node)

  def neighbors(self, idx):
    return self.node[idx].neighbors
  
  def ins(self, idx):
    return self.node[idx].ins
  
  def outs(self, idx):
    return self.node[idx].outs

  def nodes(self):
    return [n.idx for n in self.node]
