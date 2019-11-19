from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

class Operator(object):
    def __init__(self, idx, cpu, payload, weight):
        self.idx = idx
        self.cpu = cpu
        self.payload = payload
        self.weight = weight
        self.weighted_load = cpu * weight
        self.ins = []
        self.in_weights = {}
        self.outs = []
        self.out_weights = {}
        self.weights = {}
        self._build_features()
    
    def add_in(self, up_op, weight):
        self.ins.append(up_op)
        self.in_weights[up_op] = weight
        self.weights[up_op] = weight
    
    def add_out(self, down_op, weight):
        self.outs.append(down_op)
        self.out_weights[down_op] = weight
        self.weights[down_op] = weight
    
    def get_in_weight(self,indices):
        weights = []
        for i in indices:
            weights.append(self.in_weights[i])
        return weights
    
    def get_weight(self,indices):
        w = []
        for i in indices:
            w.append(self.weights[i])
        return w

    def _build_neightbors(self):
        self.neighbors = self.ins + self.outs

    def _build_features(self):
        self.feats = np.array([self.weighted_load, self.payload], dtype=np.float32)

class Graph(object):
    def __init__(self, idx, num_nodes, real_idx = -1, max_throughput = 0.0):
        self.g_id = idx
        self.num_nodes = num_nodes
        self.real_idx = real_idx
        self.node = []
        self.max_throughput = max_throughput
        self.order = []
        self.reverse_order = []
  
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


def load_data(prefix, normalize=True):
    nodes = []
    G_data = json.load(open(prefix + "-adj.json"))
    num_ops = len(G_data['ops'])
    G = Graph(num_ops)
    for op in G_data['ops']:
      node = Operator(op['idx'], op['cpu'], op['payload'], op['ins'], op['outs'])
      G.add_node(node)
      
    feats = np.vstack(n.feats for n in G.node)
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(feats)
        feats = scaler.transform(feats)
    
    return G, feats

def load_folder_data(folder, normalize = True):
    graphs = []
    all_feats = []
    for filename in os.listdir(folder):
        if filename.endswith(".json") and filename.startswith("graph"):
            nodes = []
            G_data = json.load(open(os.path.join(folder, filename)))
            g_idx = int(filename.split('.json')[0].split('_')[1])
            num_ops = len(G_data['ops'])
            G = Graph(len(graphs), num_ops, real_idx = g_idx)
            for op in G_data['ops']:
              node = Operator(op['idx'], op['cpu'], op['payload'], op['ins'], op['outs'])
              G.add_node(node)
            
            graphs.append(G) 
            feats = np.vstack([np.log(n.feats+0.00000001) for n in G.node])
            all_feats.append(feats)
    
    feats = np.vstack(_ for _ in all_feats)
    
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(feats)
        for idx, f in enumerate(all_feats):
            all_feats[idx] = scaler.transform(f)
    
    max_graph_size = max([len(G.nodes()) for G in graphs])
    for idx, f in enumerate(all_feats):
      num_nodes = f.shape[0]
      for _ in range(max_graph_size - num_nodes):
        f = np.vstack([f, np.zeros((f.shape[1],))])
      all_feats[idx] = f
    
    return graphs, all_feats

def load_ordered_folder_data(folder, normalize = True):
    graph_map = {}
    graphs = []
    all_feats = []
    num_vms = 0
    for filename in os.listdir(folder):
        if filename.endswith(".json") and filename.startswith("graph"):
            nodes = []
            G_data = json.load(open(os.path.join(folder, filename)))
            g_idx = int(filename.split('.json')[0].split('_')[1])
            num_ops = len(G_data['operators'])
            max_throughput = float(G_data['max_throughput'])
            num_vms = int(G_data['num_vms'])
            G = Graph(g_idx, num_ops, real_idx = g_idx, max_throughput = max_throughput)
            for op in G_data['operators']:
              node = Operator(op['idx'], op['cpu'], op['payload'], op['weight'])
              G.add_node(node)
            
            for edge in G_data['connections']:
              from_vertex = edge['from_vertex']
              to_vertex = edge['to_vertex']
              weight = edge['weight']
              G.node[to_vertex].add_in(from_vertex, weight)
              G.node[from_vertex].add_out(to_vertex, weight)
              
            for op in G.node:
              op._build_neightbors()

            graph_map[g_idx] = G
    
    num_graphs = len(graph_map)
    for idx in range(num_graphs):
      G = graph_map[idx]
      feats = np.vstack([np.log(n.feats+0.00000001) for n in G.node])
      all_feats.append(feats)
      graphs.append(G)

    feats = np.vstack(_ for _ in all_feats)
    
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(feats)
        for idx, f in enumerate(all_feats):
            all_feats[idx] = scaler.transform(f)
    
    max_graph_size = max([len(G.nodes()) for G in graphs])
    for idx, f in enumerate(all_feats):
      num_nodes = f.shape[0]
      for _ in range(max_graph_size - num_nodes):
        f = np.vstack([f, np.zeros((f.shape[1],))])
      all_feats[idx] = f
    
    return graphs, all_feats, num_vms

def load_sorted_folder_data(folder, normalize = True):
    graph_map = {}
    graphs = []
    all_feats = []
    orders = []
    reverse_orders = []
    num_vms = 0
    for filename in os.listdir(folder):
        if filename.endswith(".json") and filename.startswith("graph"):
            G_data = json.load(open(os.path.join(folder, filename)))
            g_idx = int(filename.split('.json')[0].split('_')[1])
            num_ops = len(G_data['operators'])
            max_throughput = float(G_data['max_throughput'])
            num_vms = int(G_data['num_vms'])
            G = Graph(g_idx, num_ops, real_idx = g_idx, max_throughput = max_throughput)
            for op in G_data['operators']:
              node = Operator(op['idx'], op['cpu'], op['payload'], op['weight'])
              G.add_node(node)
            
            for edge in G_data['connections']:
              from_vertex = edge['from_vertex']
              to_vertex = edge['to_vertex']
              weight = edge['weight']
              G.node[to_vertex].add_in(from_vertex, weight)
              G.node[from_vertex].add_out(to_vertex, weight)
              
            for op in G.node:
              op._build_neightbors()
            
            op_cpus = [o.weighted_load*-1 for o in G.node]
            G.order = np.argsort(op_cpus).tolist()
            G.reverse_order = [None] * G.num_nodes
            for i, idx in enumerate(G.order):
              G.reverse_order[idx] = i
            graph_map[g_idx] = G
    
    num_graphs = len(graph_map)
    for idx in range(num_graphs):
      G = graph_map[idx]
      feats = np.vstack([np.log(n.feats+0.00000001) for n in G.node])
      all_feats.append(feats)
      graphs.append(G)
      orders.append(G.order)
      reverse_orders.append(G.reverse_order)

    feats = np.vstack(_ for _ in all_feats)
    
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(feats)
        for idx, f in enumerate(all_feats):
            all_feats[idx] = scaler.transform(f)
    
    max_graph_size = max([len(G.nodes()) for G in graphs])
    for idx, f in enumerate(all_feats):
      num_nodes = f.shape[0]
      for _ in range(max_graph_size - num_nodes):
        f = np.vstack([f, np.zeros((f.shape[1],))])
      all_feats[idx] = f
    
    return graphs, orders, reverse_orders, all_feats, num_vms

if __name__ == "__main__":
    G, feats = load_folder_data("../exp")
