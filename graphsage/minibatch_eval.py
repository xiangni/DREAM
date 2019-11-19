from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)

class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    placeholders -- standard tensorflow placeholders object for feeding
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, placeholders, 
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.placeholders = placeholders
        self.max_degree = max_degree
        self.train_nodes = self.nodes

    def batch_feed_dict(self, batch_nodes, val=False):
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_nodes)})
        feed_dict.update({self.placeholders['batch']: batch_nodes})

        return feed_dict

    def next_minibatch_feed_dict(self):
        #feed all at a time for RL training
        start_idx = 0 
        end_idx = len(self.train_nodes)
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0


class GraphMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over graphs for reinfocement learning.

    G -- json graph
    placeholders -- standard tensorflow placeholders object for feeding
    batch_size -- size of the graph minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, graphs, placeholders, seed = 1, upstream_devices_num=5,  train_ratio = 0.8, batch_size =
    32, max_degree=25, **kwargs):

        self.graphs = graphs
        
        self.placeholders = placeholders
        
        self.upstream_devices_num = upstream_devices_num

        self.batch_size = batch_size
        self.max_degree = max_degree
        self.graph_num = 0
        self.eval_graph_num = 0

        self.adj_ins, self.adj_outs, self.deg = self.construct_adj()
        
        self.construct_dep_sources()

        np.random.seed(seed)

        num_trains = (int)(train_ratio * len(self.graphs))
        self.train_graphs = list(np.random.choice(self.graphs, num_trains, replace=False))
        print([g.real_idx for g in self.train_graphs])
        train_sets = set([g.real_idx for g in self.train_graphs])
        self.eval_graphs = []
        for g in self.graphs:
          if g.real_idx not in train_sets:
            self.eval_graphs.append(g)

    def end(self):
        return self.graph_num >= len(self.train_graphs)
    
    def eval_end(self):
        if self.eval_graph_num == len(self.eval_graphs):
          self.eval_graph_num = 0
          return True
        else:
          return False
    
    def next_batch_size(self):
        remaining = len(self.train_graphs) - self.graph_num
        return min(remaining, self.batch_size)

    def construct_adj(self):
        max_graph_size = max([len(G.nodes()) for G in self.graphs])
        
        adj_ins = max_graph_size*np.ones((len(self.graphs), max_graph_size+1, self.max_degree), dtype = np.int32)
        adj_outs = max_graph_size*np.ones((len(self.graphs), max_graph_size+1, self.max_degree), dtype = np.int32)
        
        deg = np.zeros((len(self.graphs), max_graph_size,), dtype = np.int32)
        
        for G in self.graphs:
            for nodeid in G.nodes():
                neighbors = np.array([_ for _ in G.neighbors(nodeid)])
                deg[G.g_id, nodeid] = len(neighbors)
                if len(neighbors) == 0:
                    continue
                
                ins = np.array([_ for _ in G.ins(nodeid)])
                outs = np.array([_ for _ in G.outs(nodeid)])
                
                if len(ins) != 0:
                  if len(ins) >= self.max_degree:
                      ins = np.random.choice(ins, self.max_degree, replace=False)
                  elif len(ins) < self.max_degree:
                      ins = np.random.choice(ins, self.max_degree, replace=True)
                  adj_ins[G.g_id, nodeid, :] = ins
                
                if len(outs) != 0:
                  if len(outs) >= self.max_degree:
                      outs = np.random.choice(outs, self.max_degree, replace=False)
                  elif len(outs) < self.max_degree:
                      outs = np.random.choice(outs, self.max_degree, replace=True)
                  adj_outs[G.g_id, nodeid, :] = outs
        
        return adj_ins, adj_outs, deg
    
    def construct_dep_sources(self):
        deps = []
        source_counts = []
        source_weights = []
        for idx, G in enumerate(self.graphs):
            print("processing graph {} id {}".format(idx, G.g_id))
            nodes = G.nodes()
            node_sources = np.zeros((len(nodes), self.upstream_devices_num), dtype=int)
            node_source_weights = np.zeros((len(nodes), self.upstream_devices_num), dtype=float)
            node_num_sources = np.zeros((len(nodes)), dtype=int)
            for nodeid in nodes:
                ins = G.ins(nodeid)
                
                node_num_sources[nodeid] = min(len(ins), self.upstream_devices_num)

                if len(ins) > self.upstream_devices_num:
                    ins = np.random.choice(np.array(ins), self.upstream_devices_num, replace=False)
                    node_sources[nodeid, :] = ins
                    node_source_weights[nodeid, :] = np.array(G.node[nodeid].get_in_weight(ins.tolist()))
                else:
                    node_sources[nodeid, :len(ins)] = np.array(ins)
                    node_source_weights[nodeid, :len(ins)] = np.array(G.node[nodeid].get_in_weight(ins), dtype = np.float32)
            
            deps.append(node_sources)
            source_weights.append(node_source_weights)
            source_counts.append(node_num_sources)
        
        self.deps = deps
        self.source_counts = source_counts
        self.source_weights = source_weights
  
    def batch_feed_dict(self, batch_nodes, cpu_weights, sources, source_weights, num_sources, graph_idx, real_idx, max_throughput, val=False):
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_nodes)})
        feed_dict.update({self.placeholders['batch'] : batch_nodes})
        feed_dict.update({self.placeholders['batch_sources'] : sources})
        feed_dict.update({self.placeholders['batch_source_weights'] : source_weights})
        feed_dict.update({self.placeholders['batch_num_sources'] : num_sources})
        feed_dict.update({self.placeholders['graph_idx']: graph_idx})
        self.graph_num += 1
        return feed_dict, batch_nodes, cpu_weights, len(batch_nodes), sources, source_weights, num_sources, graph_idx, real_idx, max_throughput

    def next_minibatch_feed_dict(self):
        #feed all at a time for RL training
        G = self.train_graphs[self.graph_num]
        nodes = G.nodes()
        cpu_weights = [n.weighted_load for n in G.node]
        return self.batch_feed_dict(nodes, cpu_weights, self.deps[G.g_id], self.source_weights[G.g_id], self.source_counts[G.g_id], G.g_id, G.real_idx, G.max_throughput)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_graphs = np.random.permutation(self.train_graphs)
        self.graph_num = 0
        
    def eval_batch_feed_dict(self, batch_nodes, cpu_weights, sources, source_weights, num_sources, graph_idx, real_idx, max_throughput, val=False):
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_nodes)})
        feed_dict.update({self.placeholders['batch_sources'] : sources})
        feed_dict.update({self.placeholders['batch_source_weights'] : source_weights})
        feed_dict.update({self.placeholders['batch_num_sources'] : num_sources})
        feed_dict.update({self.placeholders['batch'] : batch_nodes})
        feed_dict.update({self.placeholders['graph_idx']: graph_idx})
        self.eval_graph_num += 1
        return feed_dict, batch_nodes, cpu_weights, len(batch_nodes), sources, source_weights, num_sources, graph_idx, real_idx, max_throughput

    def next_eval_minibatch_feed_dict(self):
        #feed all at a time for RL training
        G = self.eval_graphs[self.eval_graph_num]
        nodes = G.nodes()
        cpu_weights = [n.weighted_load for n in G.node]
        return self.eval_batch_feed_dict(nodes, cpu_weights, self.deps[G.g_id], self.source_weights[G.g_id], self.source_counts[G.g_id], G.g_id, G.real_idx, G.max_throughput)
