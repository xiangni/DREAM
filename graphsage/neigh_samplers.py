from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_ins, adj_outs, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_ins = adj_ins
        self.adj_outs = adj_outs
    
    #TODO graph index included
    def _call(self, inputs):
      g_id, ids, num_samples, ins = inputs
      num_nodes = self.adj_ins.shape[1]
      degrees = self.adj_ins.shape[2]
      if ins:
        cur_adj_info = tf.slice(self.adj_ins, [g_id, 0, 0], [1, -1, -1])
        cur_adj_info = tf.reshape(cur_adj_info, [num_nodes, degrees])
        adj_lists = tf.nn.embedding_lookup(cur_adj_info, ids) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists
      else:        
        cur_adj_info = tf.slice(self.adj_outs, [g_id, 0, 0], [1, -1, -1])
        cur_adj_info = tf.reshape(cur_adj_info, [num_nodes, degrees])
        adj_lists = tf.nn.embedding_lookup(cur_adj_info, ids) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists
