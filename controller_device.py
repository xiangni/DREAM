from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import math
import random
import numpy as np
import six
import json
import os
import re
import sys
from subprocess import call, Popen, PIPE
import subprocess
from time import gmtime, strftime
import time
import pickle
import os
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops

import replay_buffer
from environment import Environment, Sample

from graphsage.utils import load_ordered_folder_data
from graphsage.minibatch_eval import GraphMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.models import SAGEInfo
from graphsage.supervised_models import SupervisedGraphsage

class PlacerParams(object):
  def __init__(self, **kwargs):
    for name, value in six.iteritems(kwargs):
      self.add_param(name, value)

  def add_param(self, name, value):
    setattr(self, name, value)

def allocator_hparams():
  """Hyperparameters for resource allocator."""
  return PlacerParams(
      hidden_size=512,
      forget_bias_init=1.0,
      grad_bound=1.0,
      lr=0.01,
      lr_dec=1.0,
      decay_steps=50,
      start_decay_step=400,
      optimizer_type="adam",
      name="hierarchical_controller",
      keep_prob=1.0,
      seed=1,
      model_size = 'small', 
      random_prob = 1.0,
      max_degree = 100,
      epoches = 10000, 
      dropout = 0.0, 
      n_explore_samples = 20,
      n_replay_samples = 5,
      replay_greedy_sampling = True,
      n_policy_samples = 10,
      train_ratio = 0.8, 
      restore = False,
      checkpoint = True,
      checkpoint_folder = 'checkpoints',
      cep_program = 'LogProcessing',
      graphsage_model = 'graphsage_maxpool', 
      samples_1 = 4,
      samples_2 = 4,
      samples_3 = 0,
      samples_4 = 0,
      samples_5 = 0,
      dim_1 = 128,
      dim_2 = 128,
      strategy = 'policy', # or memory
      replay_weight = 10.0, 
      env_batch_size = 1, # number of env in a batch
      embedding = 'original', #original or use graphsage
      feat_size = 1, #only valid if embedding is original
      decoder = 'lstm', # only lstm is supported
      placement_file = 'tmp.json',
      consider_neighbor_placement = False, #whether to consider the placement of neighbor nodes
      consider_device_utilization = False,
      weighed_neighbor_placement = True, #only useful if consider_neighbor_placement is True
      utilization_max = 2.0,
      device_scheme = 0,
      real_baseline = True,
      pool_size = 5,
      num_devices = 5,
      metis_placement = None)

class Task(object):
  def __init__(self, num_ops, graph_idx, action, device_utilizations, is_policy, placement_file, cep_program, exp_folder):
    self.num_ops = num_ops
    self.graph_idx = graph_idx
    self.action = action
    self.device_utilizations = device_utilizations
    self.is_policy = is_policy
    self.placement_file = placement_file
    self.cep_program = cep_program
    self.exp_folder = exp_folder
  
def evaluate(t):
  action = t.action
  num_ops = t.num_ops
  graph_idx = t.graph_idx
  device_utilizations = t.device_utilizations
  is_policy = t.is_policy
  placement_file = t.placement_file
  cep_program = t.cep_program
  exp_folder = t.exp_folder
  
  configs = {}
  placements = []
  vm_placements = {}
  for i in range(num_ops):
    vm = int(action[i])
    if vm not in vm_placements:
      vm_placements[vm] = []
    vm_placements[vm].append(int(i))
  for vm, ops in vm_placements.items():
    placements.append({'idx':vm, 'ops': ops})
  configs['placements'] = placements
  
  used_vms = len(vm_placements)
  with open(placement_file, 'w') as outfile:
    json.dump(configs, outfile)
  
  seq = ['timeout', '-k','30s','-s', '9', '30s','java', 'ca.uwo.eng.sel.cepsim.example.'+cep_program, placement_file, exp_folder+'/graph_'+str(graph_idx)+'.json']
  p = Popen(seq, stdout=PIPE, stderr=PIPE)
  stdout, stderr = p.communicate()
  logs = stdout
  throughput = 0
  for line in logs.split(b'\n'):
    if b"Throughputs:" in line:
      throughput += float(line.strip().split(b':')[1])
  throughput/=used_vms
  s = Sample(action, throughput, device_utilizations, is_policy = is_policy)
  return s


class ResourceAllocator():
  """ResourceAllocator class."""
  def get_time(self):
    return strftime("%Y-%m-%d-%H:%M:%S", gmtime())

  def __init__(self, hparams, exp_folder, restore_file=0):
    """ResourceAllocator class initializer.

    Args:
      hparams: All hyper-parameters.
      exp_folder: The folder contains the dataset.
    """
    self.hparams = hparams
    self.exp_folder = exp_folder
    
    self.construct_placeholders()
    
    #graph minibatch iterator 
    self.prepare_data()
    
    #build graph model
    if self.hparams.embedding == 'graphsage':
      self.build_graph_model()
    
    self.init_variables()
    
    #build sample generation 
    self.build_generate_samples()
    
    self.build_controller()  
    
    self.init()
    
    if self.hparams.restore:
      with open(self.hparams.env_restore_file, 'rb') as f:
        self.envs = pickle.load(f)
    else:  
      self.envs = {}
    
    self.ignored_samples = 0
    self.total_samples = 0
    
    self.train()
  
  def load_metis_samples(self, g_idx, batch_size, batch_loads):
    filename = str(g_idx) + ".json"
    G_data = json.load(open(os.path.join('metis_placement', filename)))
    p = G_data['placements']
    placement = {}
    for one_vm in p:
      vm_idx = int(one_vm['idx'])
      ops = one_vm['ops']
      for op in ops:
        op = int(op)
        placement[op] = vm_idx
    action = []
    for op in range(len(placement)):
      action.append(placement[op])
    action = self.uniform_action(batch_size, action)
    device_utilization = self.calculate_utilization(action, batch_loads)
    return action, device_utilization
  
  def init_variables(self):
    self.initializer = tf.glorot_uniform_initializer(seed=self.hparams.seed)
    with tf.variable_scope(
        self.hparams.name,
        initializer=self.initializer,
        reuse=tf.AUTO_REUSE):
      if self.hparams.decoder == 'lstm':
        if self.hparams.embedding == 'graphsage':
          tf.get_variable("device_softmax", [2 * self.hparams.hidden_size, self.num_devices]) #with attention
          tf.get_variable("device_embeddings", [self.num_devices, self.hparams.hidden_size])
          tf.get_variable("device_go_embedding", [1, self.hparams.hidden_size])
          tf.get_variable("attn_w_2", [self.hparams.hidden_size, self.hparams.hidden_size])
          tf.get_variable("attn_v", [self.hparams.hidden_size, 1])
          if self.hparams.consider_neighbor_placement and self.hparams.consider_device_utilization:
            w_lstm_dimension = 3 * self.hparams.hidden_size
          elif self.hparams.consider_device_utilization or self.hparams.consider_neighbor_placement:
            w_lstm_dimension = 3 * self.hparams.hidden_size
          else:  
            w_lstm_dimension = 2 * self.hparams.hidden_size
        elif self.hparams.embedding == 'original': #original embedding
          if self.hparams.consider_neighbor_placement and self.hparams.consider_device_utilization:
            w_lstm_dimension = 3 * self.hparams.hidden_size
          elif self.hparams.consider_device_utilization or self.hparams.consider_neighbor_placement:
            w_lstm_dimension = 3 * self.hparams.hidden_size
          else:  
            w_lstm_dimension = 2 * self.hparams.hidden_size
          tf.get_variable("device_softmax", [self.hparams.hidden_size, self.num_devices])
          tf.get_variable("node_embedding", [self.hparams.feat_size, self.hparams.hidden_size])
        else:
          raise NotImplementedError
        tf.get_variable("decoder_lstm", [w_lstm_dimension, 4 * self.hparams.hidden_size])
        tf.get_variable("device_aggregator", [self.num_devices, self.hparams.hidden_size])
        tf.get_variable("decoder_forget_bias", shape=1, dtype=tf.float32, initializer=tf.constant_initializer( self.hparams.forget_bias_init))
        if self.hparams.consider_device_utilization:
          tf.get_variable("device_utilization", [2*self.num_devices, self.num_devices])
      device_indices = []
      for i in range(self.num_devices):
        device_indices.append(i)
      self.device_encoding = tf.one_hot(device_indices, self.num_devices)  
  
  def init(self):
    self.sess = tf.Session()

    if self.hparams.restore:
      self.saver = tf.train.Saver()
      self.saver.restore(self.sess, self.hparams.model_restore_file)
    else:  
      init_g = tf.global_variables_initializer() 
      init_l = tf.local_variables_initializer()
      self.saver = tf.train.Saver()
      self.sess.run([init_g, init_l])
  
  def get_random_action(self, batch_size, feed_dict, batch_loads):
    a = random.randint(0, math.pow(self.num_devices, batch_size) - 1)
    action = []
    for op in range(batch_size):
      action.append(a%self.num_devices)
      a = a/self.num_devices
    action = self.uniform_action(batch_size, action)
    device_utilization_backward = self.calculate_utilization(action, batch_loads)
    return action, device_utilization_backward
  
  def get_sample_action(self, sample_size, batch_size, feed_dict, batch_loads, batch_sources, batch_num_sources, batch_source_weights, mode='sample'):
    num_children = sample_size
    feed_dict.update({self.placeholders['sample_size']:sample_size})
    feed_dict.update({self.placeholders['num_samples']:sample_size})
    device_utilization = np.zeros((num_children, self.num_devices), dtype=float) 
    prev_c = np.zeros((num_children, self.hparams.hidden_size), dtype=float)
    prev_h = np.zeros((num_children, self.hparams.hidden_size), dtype=float)
    prev_y = np.zeros((num_children), dtype=float)
    
    for i in range(num_children):
      for j in range(self.num_devices):
        device_utilization[i, j] = self.hparams.utilization_max
    
    action = [None] * sample_size
    for i in range(sample_size):
      action[i] = []

    for op in range(batch_size):
      feed_dict.update({self.placeholders['target_op']:op})
      feed_dict.update({self.placeholders['prev_c']:prev_c})
      feed_dict.update({self.placeholders['prev_h']:prev_h})
      feed_dict.update({self.placeholders['prev_y']:prev_y})
      feed_dict.update({self.placeholders['device_utilizations_forward']:device_utilization})
      if self.hparams.consider_neighbor_placement:
        source_weights = []
        source_devices = []
        for j in range(sample_size):
          source_devices.append([])
        node_sources_num = batch_num_sources[op]
        for i in range(node_sources_num):
          source_weights.append(batch_source_weights[op, i])
        for j in range(sample_size):
          for i in range(node_sources_num):
            source = batch_sources[op, i]
            source_devices[j].append(action[j][source])
        feed_dict.update({self.placeholders['source_weights']:source_weights})
        feed_dict.update({self.placeholders['source_devices']:source_devices})
        feed_dict.update({self.placeholders['node_sources_num']:node_sources_num})
        
      if mode == 'sample':
        prev_y, prev_c, prev_h = self.sess.run([self.policy_action, self.policy_c, self.policy_h], feed_dict=feed_dict)
      elif mode == 'greedy':  
        prev_y, prev_c, prev_h = self.sess.run([self.greedy_action, self.greedy_c, self.greedy_h], feed_dict=feed_dict)
      else:
        raise NotImplementedError
      for i in range(sample_size):
        action[i].append(prev_y[i])
        load = batch_loads[op]/1e6
        device_utilization[i, action[i][op]] -= load
    
    uniform_actions = []
    device_utilization_backward = []
    for i in range(sample_size):
      a = self.uniform_action(batch_size, action[i])
      uniform_actions.append(a)
      device_utilization_backward.append(self.calculate_utilization(a, batch_loads))
    return uniform_actions, device_utilization_backward

  def get_action(self, batch_size, feed_dict, batch_loads, batch_sources, batch_num_sources, batch_source_weights, mode='sample'):
    num_children = 1
    feed_dict.update({self.placeholders['num_samples']:1})
    device_utilization = np.zeros((num_children, self.num_devices), dtype=float) 
    prev_c = np.zeros((num_children, self.hparams.hidden_size), dtype=float)
    prev_h = np.zeros((num_children, self.hparams.hidden_size), dtype=float)
    prev_y = np.zeros((num_children), dtype=float)
    
    for i in range(num_children):
      for j in range(self.num_devices):
        device_utilization[i, j] = self.hparams.utilization_max
    
    action = []
    for op in range(batch_size):
      feed_dict.update({self.placeholders['target_op']:op})
      feed_dict.update({self.placeholders['prev_c']:prev_c})
      feed_dict.update({self.placeholders['prev_h']:prev_h})
      feed_dict.update({self.placeholders['prev_y']:prev_y})
      feed_dict.update({self.placeholders['device_utilizations_forward']:device_utilization})
      if self.hparams.consider_neighbor_placement:
        source_weights = []
        source_devices = []
        for j in range(1):
          source_devices.append([])
        node_sources_num = batch_num_sources[op]
        for i in range(node_sources_num):
          source_weights.append(batch_source_weights[op, i])
        for j in range(1):
          for i in range(node_sources_num):
            source = batch_sources[op, i]
            source_devices[j].append(action[source])
        feed_dict.update({self.placeholders['source_weights']:source_weights})
        feed_dict.update({self.placeholders['source_devices']:source_devices})
        feed_dict.update({self.placeholders['node_sources_num']:node_sources_num})
        
      if mode == 'sample':
        prev_y, prev_c, prev_h = self.sess.run([self.policy_action, self.policy_c, self.policy_h], feed_dict=feed_dict)
      elif mode == 'greedy':  
        prev_y, prev_c, prev_h = self.sess.run([self.greedy_action, self.greedy_c, self.greedy_h], feed_dict=feed_dict)
      else:
        raise NotImplementedError
      action.append(prev_y[0])
      load = batch_loads[op]/1e6
      device_utilization[0, action[op]] -= load
    
    action = self.uniform_action(batch_size, action)
    device_utilization_backward = self.calculate_utilization(action, batch_loads)
    return action, device_utilization_backward
  
  def get_sample(self, batch_size, real_idx, action, device_utilization, g_idx = -1):
    action_str = ''.join(str(i) for i in action)
    s=None
    if g_idx !=-1:
      throughput = self.envs[g_idx].if_exist(action_str)
      if throughput == -1:
        throughput = self.evaluate_cepsim(batch_size, real_idx, action)
    else: 
      throughput = self.evaluate_cepsim(batch_size, real_idx, action)
    if throughput > 0:
      s = Sample(action, throughput, device_utilization)
    return s

  def prepare_samples_for_back(self, fd, train_samples, epoch, num_replay_samples):  
    if len(train_samples) == 0:
      return None
    actions = np.vstack([s.action for s in train_samples])
    ranks = [s.rank for s in train_samples]
    utilizations = [s.device_utilization for s in train_samples]
    utilizations = np.concatenate(utilizations, axis=1)
    fd['device_utilizations'] = utilizations
    if self.hparams.real_baseline == True:
      baseline = self.envs[fd['graph_idx']].calculate_baseline(epoch, num_replay_samples)
    else:
      baseline = np.mean(ranks)
    
    probs = self.compute_probs(actions, actions.shape[0], fd)
    for idx, r in enumerate(ranks):
      r_origin = r
      r -= baseline
      ranks[idx] = r
      p  = probs[idx]
      if p == 0.0 and r < 0: #if negative sample and probability is already small, ignore it
        ranks[idx] = 0
        self.ignored_samples += 1
      else:
        self.total_samples += 1
      if r > 0:
        ranks[idx] = self.hparams.replay_weight * r
    
    train_ranks = np.array(ranks)
    fd['num_actions'] = len(train_samples)
    fd['actions'] = actions
    fd['reward'] = train_ranks
    return fd

  def train(self):
    for epoch in range(self.hparams.epoches):
      epoch_start_time = time.time()
      self.minibatch.shuffle()
      print("start epoch {} {}".format(epoch, epoch_start_time))
      self.ignored_samples = 0
      self.total_samples = 0
      while not self.minibatch.end():
        one_start_time = time.time()
        dict_for_back = {}
        graph_batch_size = self.minibatch.next_batch_size()
        for local_idx in range(graph_batch_size):
          feed_dict, batch, batch_loads, batch_size, batch_sources, batch_source_weights, num_batch_sources, graph_idx, real_idx, max_throughput = self.minibatch.next_minibatch_feed_dict()
          feed_dict.update({self.placeholders['dropout']: self.hparams.dropout})
          feed_dict.update({self.placeholders['sample_size']: 1})
          fd = {'batch_size': batch_size, 'batch' : batch, 'batch_sources' : batch_sources, 'batch_num_sources':num_batch_sources, 'graph_idx': graph_idx, 'batch_source_weights' : batch_source_weights}
          
          if graph_idx not in self.envs:
            self.envs[graph_idx] = Environment(graph_idx, batch_size, max_throughput, queue_leangth = 30)
          
          if self.hparams.strategy == 'policy':
            for _ in range(self.hparams.n_policy_samples):
              if np.random.rand() < self.hparams.random_prob/np.exp(epoch):
                action, device_utilization = self.get_random_action(batch_size, feed_dict, batch_loads)
              else:
                action, device_utilization = self.get_action(batch_size, feed_dict, batch_loads, batch_sources, num_batch_sources, batch_source_weights, mode = 'sample')
              
              s = self.get_sample(batch_size, real_idx, action, device_utilization, g_idx=graph_idx)
              if s != None:
                self.envs[graph_idx].save(s, on_policy=True, build_replay=False)
              
            policy_samples = self.envs[graph_idx].sample(self.hparams.n_policy_samples) 
            
          elif self.hparams.strategy == 'memory':
            random_prob = self.hparams.random_prob/np.exp(epoch)
            num_random_samples = (int)(self.hparams.n_explore_samples * random_prob)
            if self.envs[graph_idx].hard_problem():
              num_random_samples = self.hparams.n_explore_samples 
            tasks = []
            random_set = set()
            for _ in range(num_random_samples): 
              action, device_utilization = self.get_random_action(batch_size, feed_dict, batch_loads)
              action_str = ''.join(str(i) for i in action)
              if action_str not in random_set:
                random_set.add(action_str)
                throughput = self.envs[graph_idx].if_exist(action_str) 
                if throughput != -1:
                  s = Sample(action, throughput, device_utilization)
                  self.envs[graph_idx].save(s)
                else:
                  t = Task(batch_size, real_idx, action, device_utilization, False, self.hparams.placement_file+'_'+str(len(tasks))+'.json', self.hparams.cep_program, self.exp_folder)
                  tasks.append(t)
            #load metis placement
            if epoch == 0 and self.hparams.metis_placement != None:
              action, device_utilization = self.load_metis_samples(real_idx, batch_size, batch_loads)
              t = Task(batch_size, real_idx, action, device_utilization, False, self.hparams.placement_file+'_'+str(len(tasks))+'.json', self.hparams.cep_program, self.exp_folder)
              tasks.append(t)
            policy_samples = []
            policy_set = set()
            start_policy_time = time.time();
            actions, device_utilizations = self.get_sample_action(self.hparams.n_policy_samples, batch_size, feed_dict, batch_loads, batch_sources, num_batch_sources, batch_source_weights, mode = 'sample')
            for p in range(self.hparams.n_policy_samples): 
              action_str = ''.join(str(i) for i in actions[p])
              if action_str not in policy_set:
                policy_set.add(action_str)
                throughput = self.envs[graph_idx].if_exist(action_str)
                if throughput == -1:
                  t = Task(batch_size, real_idx, actions[p], device_utilizations[p], True, self.hparams.placement_file+'_'+str(len(tasks))+'.json', self.hparams.cep_program, self.exp_folder)
                  tasks.append(t)
                else:
                  s = Sample(actions[p], throughput, device_utilizations[p])
                  self.envs[graph_idx].save(s, on_policy = True)
                  policy_samples.append(s)
            
            
            start_time = time.time()
            with ThreadPool(min(10, max(self.hparams.pool_size, len(tasks)))) as p:
              pending_samples = p.map(evaluate, tasks)
              for s in pending_samples:
                if s.is_policy:
                  self.envs[graph_idx].save(s, on_policy = True)
                  policy_samples.append(s)
                else:
                  self.envs[graph_idx].save(s)

            replay_samples = self.envs[graph_idx].replay(self.hparams.n_replay_samples, greedy = self.hparams.replay_greedy_sampling)
            train_samples = replay_samples + policy_samples
            
            dict_for_back = self.prepare_samples_for_back(fd, train_samples, epoch, len(replay_samples))
          else:    
            raise NotImplementedError
        if dict_for_back != None:
          self.optimize(dict_for_back)
      
      start_time = time.time()
      self.test_w_throughput()
      
      if self.hparams.checkpoint and epoch % 10 == 0:
        self.save(epoch)
  
  def test_w_throughput(self):
    tasks = []
    while not self.minibatch.eval_end():
      feed_dict, batch, batch_loads, batch_size, batch_sources, batch_source_weights, num_batch_sources, graph_idx, real_idx, max_throughput = self.minibatch.next_eval_minibatch_feed_dict()
      if graph_idx not in self.envs:
        self.envs[graph_idx] = Environment(graph_idx, batch_size, max_throughput)

      feed_dict.update({self.placeholders['dropout']: self.hparams.dropout})
      feed_dict.update({self.placeholders['sample_size']: 1})
      action, _ = self.get_action(batch_size, feed_dict, batch_loads, batch_sources, num_batch_sources, batch_source_weights, mode = 'greedy')
      action_str = ''.join(str(i) for i in action)
      throughput = self.envs[graph_idx].if_exist(action_str) 
      if throughput != -1:
        print("evaluating graph {}, greedy placement".format(graph_idx))
        print("action {} rank {}".format(action_str, throughput/max_throughput))
      else:
        t = Task(batch_size, real_idx, action, None, False, self.hparams.placement_file+'_'+str(len(tasks))+'.json', self.hparams.cep_program, self.exp_folder)
        tasks.append(t)

      if len(tasks) == 10:
        with Pool(10) as p:
          samples = p.map(evaluate, tasks)
          for s, t in zip(samples, tasks):
            r = s.throughput/self.envs[t.graph_idx].max_throughput
            self.envs[t.graph_idx].save_test(s.throughput, s.action_str)
            print("evaluating graph {}, greedy placement".format(t.graph_idx))
            print("action {} rank {}".format(s.action_str, r))
          tasks = []
    
    if len(tasks) > 0:  
      with Pool(min(10, len(tasks))) as p:
        samples = p.map(evaluate, tasks)
        for s, t in zip(samples, tasks):
          r = s.throughput/self.envs[t.graph_idx].max_throughput
          self.envs[t.graph_idx].save_test(s.throughput, s.action_str)
          print("evaluating graph {}, greedy placement".format(t.graph_idx))
          print("action {} rank {}".format(s.action_str, r))
        tasks = []
  
  def exec_no_fail(self,seq):
    p = Popen(seq, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    return stdout
  
  def uniform_action(self, num_ops, action):
    mapped_action = {}
    op_to_vm = {}
    vm_idx = 0
    for i in range(num_ops):
      vm = int(action[i])
      if vm not in mapped_action:
        mapped_action[vm] = vm_idx
        vm_idx += 1
      op_to_vm[int(i)] = mapped_action[vm]
    
    new_actions = []
    for op, vm in op_to_vm.items():
      new_actions.append(vm)

    return new_actions

  def calculate_utilization(self, action, batch_loads):
    device_utilization = []
    current_utilization = np.zeros((1, self.num_devices), dtype=float)
    for i in range(1):
      for j in range(self.num_devices):
        current_utilization[i, j] = self.hparams.utilization_max
    
    device_utilization.append(current_utilization)
    for idx, op in enumerate(batch_loads[:-1]):
      utilization = np.copy(current_utilization)
      utilization[0, action[idx]] -= batch_loads[idx]/1e6   
      device_utilization.append(utilization)
      current_utilization = utilization
    device_utilization = np.array(device_utilization)
    return device_utilization

  def evaluate_cepsim(self, num_ops, graph_idx, action):
    configs = {}
    placements = []
    
    vm_placements = {}
    for i in range(num_ops):
      vm = int(action[i])
      if vm not in vm_placements:
        vm_placements[vm] = []
      vm_placements[vm].append(int(i))
    for vm, ops in vm_placements.items():
      placements.append({'idx':vm, 'ops': ops})
    configs['placements'] = placements
    
    used_vms = len(vm_placements)
    with open(self.hparams.placement_file, 'w') as outfile:
      json.dump(configs, outfile)
      outfile.flush()
      os.fsync(outfile.fileno())
      outfile.close()
    
    logs = self.exec_no_fail(['timeout', '-k','30s','-s', '9', '30s','java', 'ca.uwo.eng.sel.cepsim.example.'+self.hparams.cep_program, self.hparams.placement_file, self.exp_folder+'/graph_'+str(graph_idx)+'.json'])
    throughput = 0
    for line in logs.split(b'\n'):
      if b"Throughputs:" in line:
        throughput += float(line.strip().split(b':')[1])
    return throughput/used_vms
    
  def save(self, epoch):
    save_path = self.saver.save(self.sess, self.hparams.checkpoint_folder + '/mode.'+str(epoch)+".ckpt")
    print("Model saved in path: %s" % save_path)
    #store the environment as well
    with open(self.hparams.checkpoint_folder +'/' + str(epoch) + '.pkl', 'wb') as f:
      pickle.dump(self.envs, f, pickle.HIGHEST_PROTOCOL)

  def construct_placeholders(self):
    placeholders = {
      'sample_size' : tf.placeholder(tf.int32, name='sample_size'),
      'num_samples' : tf.placeholder(tf.int32, name='num_samples'),
      'dropout' : tf.placeholder_with_default(0., shape=(), name='dropout'),
      'random_devices_logits' : tf.placeholder(tf.float32, shape=(None, None), name='random_devices_logits'),
      'reward' : tf.placeholder(tf.float32, shape=(None), name='reward'),
      'actions' : tf.placeholder(tf.int32, shape=(None, None),
      name='sample_actions'),
      'num_actions' : tf.placeholder(tf.int32, name = 'num_actions'),
      'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
      'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
      'graph_idx' : tf.placeholder(tf.int32, name='graph_idx'),
      'batch_sources' : tf.placeholder(tf.int32, shape = (None, None), name = 'batch_sources'),
      'batch_num_sources' : tf.placeholder(tf.int32, shape = (None), name = 'batch_num_sources'),
      'batch_source_weights' : tf.placeholder(tf.float32, shape = (None, None), name = 'batch_source_weights'),
      'device_utilizations' : tf.placeholder(tf.float32, shape = (None, None, None), name = 'device_utilization'),
      'device_utilizations_forward' : tf.placeholder(tf.float32, shape = (None, None), name = 'device_utilization_forward'),
      'target_op': tf.placeholder(tf.int32, name = 'target_op'),
      'prev_y' : tf.placeholder(tf.int32, shape = (None), name = 'prev_y'),
      'prev_c' : tf.placeholder(tf.float32, shape = (None, None), name = 'prev_c'),
      'prev_h' : tf.placeholder(tf.float32, shape = (None, None), name = 'prev_h'),
      'source_weights' : tf.placeholder(tf.float32, shape = (None), name = 'source_weights'),
      'source_devices' : tf.placeholder(tf.int32, shape = (None, None), name = 'source_devices'),
      'node_sources_num' : tf.placeholder(tf.int32, name = 'node_sources_num'),
    }
    self.placeholders = placeholders
  
  def prepare_data(self):
    G, features, num_devices = load_ordered_folder_data(self.exp_folder)
    self.num_devices = self.hparams.num_devices
    #create Batch iterator with G and feats
    self.minibatch = GraphMinibatchIterator(G, self.placeholders, seed = self.hparams.seed, train_ratio = self.hparams.train_ratio, batch_size =
    self.hparams.env_batch_size, max_degree = self.hparams.max_degree)
    
    for idx, f in enumerate(features):
      features[idx] = np.vstack([f, np.zeros((f.shape[1],))])
    
    self.features = np.array(features) 

  def build_graph_model(self):
    sampler = UniformNeighborSampler(self.minibatch.adj_ins, self.minibatch.adj_outs)
    layer_infos = [SAGEInfo("node", sampler, self.hparams.samples_1, self.hparams.dim_1),
                  SAGEInfo("node", sampler, self.hparams.samples_2, self.hparams.dim_2)]
    if self.hparams.samples_3 != 0:
      layer_infos.append(SAGEInfo("node", sampler, self.hparams.samples_3, self.hparams.dim_2))
      if self.hparams.samples_4 != 0:
        layer_infos.append(SAGEInfo("node", sampler, self.hparams.samples_4, self.hparams.dim_2))
        if self.hparams.samples_5 != 0:
          layer_infos.append(SAGEInfo("node", sampler, self.hparams.samples_5, self.hparams.dim_2))
          
    if self.hparams.graphsage_model == 'graphsage_mean':
      self.model = SupervisedGraphsage(self.placeholders, 
                                     self.features,
                                     layer_infos, 
                                     batch_size = self.hparams.env_batch_size, 
                                     model_size=self.hparams.model_size,
                                     logging=True)
    
    elif self.hparams.graphsage_model == 'gcn':
      # Create model
      self.model = SupervisedGraphsage(self.placeholders, 
                                   self.features,
                                   layer_infos, 
                                   aggregator_type="gcn",
                                   batch_size = self.hparams.env_batch_size, 
                                   model_size=self.hparams.model_size,
                                   concat=False,
                                   logging=True)

    elif self.hparams.graphsage_model == 'graphsage_seq':
      self.model = SupervisedGraphsage(self.placeholders, 
                                   self.features,
                                   layer_infos, 
                                   aggregator_type="seq",
                                   model_size=self.hparams.model_size,
                                   batch_size = self.hparams.env_batch_size, 
                                   logging=True)

    elif self.hparams.graphsage_model == 'graphsage_maxpool':
      self.model = SupervisedGraphsage(self.placeholders, 
                                   self.features,
                                   layer_infos, 
                                   aggregator_type="maxpool",
                                   model_size=self.hparams.model_size,
                                   batch_size = self.hparams.env_batch_size, 
                                   logging=True)

    elif self.hparams.graphsage_model == 'graphsage_meanpool':
      self.model = SupervisedGraphsage(self.placeholders, 
                                  self.features,
                                  layer_infos, 
                                   aggregator_type="meanpool",
                                   model_size=self.hparams.model_size,
                                   batch_size = self.hparams.env_batch_size, 
                                   logging=True)

    else:
      raise Exception('Error: model name unrecognized.')

    
    self.node_embeddings = self.model.get_node_preds()
    self.graph_embeddings = self.model.get_graph_preds()

  def build_controller(self):
    self._global_step = tf.train.get_or_create_global_step()
    
    ctr = {}
    ctr["loss"] = 0

    actions = self.placeholders['actions']
    num_actions = self.placeholders['num_actions']
    reward = self.placeholders['reward']
    ctr["probs"] = self.get_probs(actions, num_actions)
    
    ctr["loss"] = tf.reduce_mean(reward * ctr["probs"])

    with tf.variable_scope(
        "optimizer", reuse=tf.AUTO_REUSE):
      (ctr["train_op"], ctr["lr"], ctr["grad_norm"],
      ctr["grad_norms"]) = self._get_train_ops(
           ctr["loss"],
           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
           self.global_step,
           grad_bound=self.hparams.grad_bound,
           lr_init=self.hparams.lr,
           lr_dec=self.hparams.lr_dec,
           start_decay_step=self.hparams.start_decay_step,
           decay_steps=self.hparams.decay_steps,
           optimizer_type=self.hparams.optimizer_type)

    self.ctr = ctr

  @property
  def global_step(self):
    return self._global_step

  def build_generate_samples(self):
    if self.hparams.decoder == 'lstm':
      if self.hparams.embedding == 'graphsage':
        [random_action], [policy_action, policy_c, policy_h], [greedy_action, greedy_c, greedy_h] = self.build_graphsage_lstm_decoder()
      else:
        [random_action], [policy_action, policy_c, policy_h], [greedy_action, greedy_c, greedy_h] = self.build_original_lstm_decoder()

      self.policy_action = policy_action
      self.greedy_action = greedy_action
      self.random_action = random_action
      self.greedy_c = greedy_c
      self.greedy_h = greedy_h
      self.policy_c = policy_c
      self.policy_h = policy_h
    else:
      raise NotImplementedError

  def build_graphsage_lstm_decoder(self):
    ph = self.placeholders
    sample_size = ph["sample_size"]
    random_devices_logits = ph["random_devices_logits"]
    random_action = self.encode_forward_random(random_devices_logits)
    
    attn_mem = self.node_embeddings
    last_h = self.graph_embeddings
    attn_mem = tf.expand_dims(attn_mem, 0)
    attn_mem = tf.tile(attn_mem, [sample_size, 1, 1])
    last_h = tf.expand_dims(last_h, 0)
    last_h = tf.tile(last_h, [sample_size, 1, 1])
    last_h = tf.reshape(last_h, [sample_size, self.hparams.hidden_size])
    
    device_utilization = self.placeholders['device_utilizations_forward']
    target_op = self.placeholders['target_op']
    prev_c = self.placeholders['prev_c']
    prev_h = self.placeholders['prev_h']
    prev_y = self.placeholders['prev_y']
      
    num_ops = self.placeholders['batch_size']
    policy_action, policy_c, policy_h = (self.decode_forward(num_ops, target_op, sample_size, prev_c, prev_h, last_h, attn_mem, device_utilization, prev_y, mode="sample"))
    greedy_action, greedy_c, greedy_h = (self.decode_forward(num_ops, target_op, sample_size, prev_c, prev_h, last_h, attn_mem, device_utilization, prev_y, mode="greedy"))
    return [random_action], [policy_action, policy_c, policy_h], [greedy_action, greedy_c, greedy_h]
  
  def build_original_lstm_decoder(self):
    ph = self.placeholders
    sample_size = ph["sample_size"]
    random_devices_logits = ph["random_devices_logits"]
    g_id = ph["graph_idx"]
    random_action = self.encode_forward_random(random_devices_logits)

    self.features = tf.Variable(tf.constant(self.features, dtype=tf.float32), trainable=False)
    graph_idx = self.placeholders["graph_idx"]
    input_features = tf.slice(self.features, [g_id, 0, 0], [1, -1, -1])
    input_features = tf.reshape(input_features, [self.features.shape[1], self.features.shape[2]])
    
    device_utilization = self.placeholders['device_utilizations_forward']
    target_op = self.placeholders['target_op']
    prev_c = self.placeholders['prev_c']
    prev_h = self.placeholders['prev_h']
    
    policy_action, policy_c, policy_h = (self.encode_forward(target_op, input_features, sample_size, prev_c, prev_h, device_utilization, mode='sample'))
    greedy_action, greedy_c, greedy_h = (self.encode_forward(target_op, input_features, sample_size, prev_c, prev_h, device_utilization, mode='greedy'))
    
    return [random_action], [policy_action, policy_c, policy_h], [greedy_action, greedy_c, greedy_h]


  def compute_probs(self, actions, num_actions, fd):
    feed_dict = dict()
    feed_dict.update({self.placeholders['num_actions'] : num_actions})
    feed_dict.update({self.placeholders['actions'] : actions})
    feed_dict.update({self.placeholders['batch'] : fd['batch']})
    feed_dict.update({self.placeholders['batch_sources'] : fd['batch_sources']})
    feed_dict.update({self.placeholders['batch_source_weights'] : fd['batch_source_weights']})
    feed_dict.update({self.placeholders['device_utilizations'] : fd['device_utilizations']})
    feed_dict.update({self.placeholders['batch_num_sources'] : fd['batch_num_sources']})
    feed_dict.update({self.placeholders['batch_size'] : fd['batch_size']})
    feed_dict.update({self.placeholders['graph_idx'] : fd['graph_idx']})
    probs = self.sess.run(self.ctr["probs"], feed_dict=feed_dict)
    return [np.exp(-l) for l in probs]

  def get_probs_graphsage_lstm(self, actions, num_actions):
    device_utilizations = self.placeholders['device_utilizations']
    attn_mem = self.node_embeddings
    last_h = self.graph_embeddings
    attn_mem = tf.expand_dims(attn_mem, 0)
    attn_mem = tf.tile(attn_mem, [num_actions, 1, 1])
    last_h = tf.expand_dims(last_h, 0)
    last_h = tf.tile(last_h, [num_actions, 1, 1])
    last_h = tf.reshape(last_h, [num_actions, self.hparams.hidden_size])
    
    num_ops = self.placeholders['batch_size']
    
    _, log_probs = (self.decode(num_ops, num_actions,
            last_h, attn_mem, device_utilizations, actions))
    
    return log_probs
  
  def get_probs_original_lstm(self, actions, num_actions):
    device_utilizations = self.placeholders['device_utilizations']
    num_ops = self.placeholders['batch_size']
    g_id = self.placeholders["graph_idx"]
    input_features = tf.slice(self.features, [g_id, 0, 0], [1, -1, -1])
    input_features = tf.reshape(input_features, [self.features.shape[1], self.features.shape[2]])
    
    _, log_probs = (self.encode(input_features, num_ops, num_actions, device_utilizations, actions))
    
    return log_probs

  def get_probs(self, actions, num_actions):
    if self.hparams.decoder == 'lstm':
      if self.hparams.embedding == 'graphsage':
        return self.get_probs_graphsage_lstm(actions, num_actions)
      elif self.hparams.embedding == 'original':
        return self.get_probs_original_lstm(actions, num_actions)
      else:
        raise NotImplementedError
  
  def aggregate_source_devices(self, i, actions, num_children, sources, node_sources_num, source_weights):
    with tf.variable_scope(self.hparams.name, reuse=tf.AUTO_REUSE):
      device_aggregator = tf.get_variable("device_aggregator")
    
    node_sources = tf.slice(sources, [i, 0], [1, node_sources_num])
    node_sources = tf.reshape(node_sources, [node_sources_num])
    if self.hparams.weighed_neighbor_placement:
      source_weights = tf.slice(source_weights, [i, 0], [1, node_sources_num])
      source_weights = tf.reshape(source_weights, [node_sources_num])
      source_weights = tf.expand_dims(source_weights, 0)
      source_weights = tf.tile(source_weights, [num_children, 1])
      source_weights = tf.reshape(source_weights, [num_children*node_sources_num, 1])

    source_actions = tf.map_fn(lambda x:actions.read(x), node_sources)
    source_actions = tf.transpose(source_actions, [1, 0])
    source_actions = tf.reshape(source_actions, [node_sources_num * num_children])
    source_devices = tf.nn.embedding_lookup(self.device_encoding, source_actions)
    #
    source_devices = tf.reshape(source_devices, (node_sources_num * num_children, self.num_devices))
    if self.hparams.weighed_neighbor_placement:
      source_devices = source_weights  * source_devices
    source_devices_embeddings = tf.matmul(source_devices, device_aggregator)
    source_devices_embeddings = tf.reshape(source_devices_embeddings, [num_children, node_sources_num, self.hparams.hidden_size])
    if self.hparams.weighed_neighbor_placement:
      source_devices_embeddings = tf.reduce_sum(source_devices_embeddings, axis=1)
    else:
      source_devices_embeddings = tf.reduce_mean(source_devices_embeddings, axis=1)
    
    source_devices_embeddings = tf.reshape(source_devices_embeddings, [num_children, self.hparams.hidden_size])
    
    return source_devices_embeddings
  
  def aggregate_source_devices_forward(self, num_children, source_devices, source_weights, node_sources_num):
    with tf.variable_scope(self.hparams.name, reuse=tf.AUTO_REUSE):
      device_aggregator = tf.get_variable("device_aggregator")
    
    source_devices = tf.reshape(source_devices, [num_children*node_sources_num, 1])
    source_devices = tf.nn.embedding_lookup(self.device_encoding, source_devices)
    
    source_devices = tf.reshape(source_devices, (node_sources_num * num_children, self.num_devices))
    if self.hparams.weighed_neighbor_placement:
      source_weights = tf.expand_dims(source_weights, 0)
      source_weights = tf.tile(source_weights, [num_children, 1])
      source_weights = tf.reshape(source_weights, [num_children*node_sources_num, 1])
      source_devices = source_weights  * source_devices
    source_devices_embeddings = tf.matmul(source_devices, device_aggregator)
    source_devices_embeddings = tf.reshape(source_devices_embeddings, [num_children, node_sources_num, self.hparams.hidden_size])
    if self.hparams.weighed_neighbor_placement:
      source_devices_embeddings = tf.reduce_sum(source_devices_embeddings, axis=1)
    else:
      source_devices_embeddings = tf.reduce_mean(source_devices_embeddings, axis=1)
    
    source_devices_embeddings = tf.reshape(source_devices_embeddings, [num_children, self.hparams.hidden_size])
    return source_devices_embeddings
  
  def decode_forward(self, num_ops, target_op, num_children, prev_c, prev_h, last_h, attn_mem, device_utilizations, prev_y, mode = 'sample'):
    h = tf.cond(tf.equal(target_op, 0),
        lambda: last_h,
        lambda: prev_h)
    ph = self.placeholders
    
    num_samples = self.placeholders['num_samples']

    with tf.variable_scope(self.hparams.name, reuse=tf.AUTO_REUSE):
      w_lstm = tf.get_variable("decoder_lstm")
      forget_bias = tf.get_variable("decoder_forget_bias")
      device_embeddings = tf.get_variable("device_embeddings")
      device_softmax = tf.get_variable("device_softmax")
     
      device_go_embedding = tf.get_variable("device_go_embedding")
      attn_w_2 = tf.get_variable("attn_w_2")
      attn_v = tf.get_variable("attn_v")
    
    attn = tf.slice(self.node_embeddings, [target_op, 0], [1, -1])
    attn = tf.reshape(attn, [1, self.hparams.hidden_size])
    attn = tf.expand_dims(attn, 0)
    attn = tf.tile(attn, [num_children, 1, 1])
    attn = tf.reshape(attn, [num_children, self.hparams.hidden_size])
        
    signal = attn
    
    if self.hparams.consider_neighbor_placement:
      node_sources_num = self.placeholders['node_sources_num']
      source_devices = self.placeholders['source_devices']
      source_weights = self.placeholders['source_weights']
      source_devices_embeddings = tf.cond(tf.equal(node_sources_num, 0), lambda: tf.zeros([num_children, self.hparams.hidden_size], dtype=tf.float32), lambda: self.aggregate_source_devices_forward(num_children, source_devices, source_weights, node_sources_num))
      signal = tf.concat([signal, source_devices_embeddings], axis = 1)
    
    next_c, next_h = lstm(signal, prev_c, h, w_lstm, forget_bias)
    query = tf.matmul(next_h, attn_w_2)
    query = tf.reshape(query, [num_children, 1, self.hparams.hidden_size])
    query = tf.tanh(query + attn_mem)
    query = tf.reshape(query, [num_children * num_ops, self.hparams.hidden_size])
    query = tf.matmul(query, attn_v)
    query = tf.reshape(query, [num_children, num_ops])
    query = tf.nn.softmax(query)
    query = tf.reshape(query, [num_children, num_ops, 1])
    query = tf.reduce_sum(attn_mem * query, axis=1)
    query = tf.concat([next_h, query], axis=1)
    logits = tf.matmul(query, device_softmax)
    
    if self.hparams.consider_device_utilization:
      if self.hparams.device_scheme == 0:
        with tf.variable_scope(self.hparams.name, reuse=tf.AUTO_REUSE):
          w_utilization = tf.get_variable("device_utilization")
        logits = tf.concat([logits, device_utilizations], axis = 1)
        logits = tf.matmul(logits, w_utilization)
      else:
        logits = tf.nn.softmax(logits)
        logits = logits * device_utilizations
        logits = tf.log(logits)

    if mode == "sample":
      next_y = tf.multinomial(logits, 1, seed=self.hparams.seed)
    elif mode == "greedy":
      next_y = tf.argmax(logits, 1)
    else:
      raise NotImplementedError
        
    next_y = tf.to_int32(next_y)
    next_y = tf.reshape(next_y, [num_children])
    return next_y, next_c, next_h
  
  #used for back propogation
  def decode(self,
             num_ops,
             num_children,
             last_h,
             attn_mem,
             device_utilizations, 
             y):
    ph = self.placeholders
    sources = ph["batch_sources"]
    num_sources = ph["batch_num_sources"]
    source_weights = ph["batch_source_weights"]

    with tf.variable_scope(self.hparams.name, reuse=tf.AUTO_REUSE):
      w_lstm = tf.get_variable("decoder_lstm")
      forget_bias = tf.get_variable("decoder_forget_bias")
      device_embeddings = tf.get_variable("device_embeddings")
      device_softmax = tf.get_variable("device_softmax")
     
      device_go_embedding = tf.get_variable("device_go_embedding")
      attn_w_2 = tf.get_variable("attn_w_2")
      attn_v = tf.get_variable("attn_v")
    
    actions = tensor_array_ops.TensorArray(
        tf.int32,
        size=num_ops,
        infer_shape=False,
        clear_after_read=False)
    
    def condition(i, *args):
      return tf.less(i, num_ops)

    def body(i, prev_c, prev_h, actions, log_probs):
      attn = tf.slice(self.node_embeddings, [i, 0], [1, -1])
      attn = tf.reshape(attn, [1, self.hparams.hidden_size])
      attn = tf.expand_dims(attn, 0)
      attn = tf.tile(attn, [num_children, 1, 1])
      attn = tf.reshape(attn, [num_children, self.hparams.hidden_size])
      
      signal = attn
      if self.hparams.consider_neighbor_placement:
        node_sources_num = tf.gather(num_sources, i)
        source_devices_embeddings = tf.cond(tf.equal(node_sources_num, 0), lambda: tf.zeros([num_children, self.hparams.hidden_size], dtype=tf.float32), lambda: self.aggregate_source_devices(i, actions, num_children, sources, node_sources_num, source_weights))
        signal = tf.concat([signal, source_devices_embeddings], axis = 1)
      next_c, next_h = lstm(signal, prev_c, prev_h, w_lstm, forget_bias)
      query = tf.matmul(next_h, attn_w_2)
      query = tf.reshape(
          query, [num_children, 1, self.hparams.hidden_size])
      query = tf.tanh(query + attn_mem)
      query = tf.reshape(query, [
          num_children * num_ops, self.hparams.hidden_size
      ])
      query = tf.matmul(query, attn_v)
      query = tf.reshape(query, [num_children, num_ops])
      query = tf.nn.softmax(query)
      query = tf.reshape(query, [num_children, num_ops, 1])
      query = tf.reduce_sum(attn_mem * query, axis=1)
      query = tf.concat([next_h, query], axis=1)
      logits = tf.matmul(query, device_softmax)
      if self.hparams.consider_device_utilization:
        utilization = tf.slice(device_utilizations, [i, 0, 0], [1, -1, -1])
        utilization = tf.reshape(utilization, [num_children, self.num_devices])
        if self.hparams.device_scheme == 0:
          logits = tf.concat([logits, utilization], axis = 1)
          with tf.variable_scope(self.hparams.name, reuse=tf.AUTO_REUSE):
            w_utilization = tf.get_variable("device_utilization")
          logits = tf.matmul(logits, w_utilization)
        else:
          logits = tf.nn.softmax(logits)
          logits = logits * utilization
          logits = tf.log(logits)

      next_y = tf.slice(y, [0, i], [-1, 1])
        
      next_y = tf.to_int32(next_y)
      next_y = tf.reshape(next_y, [num_children])
      actions = actions.write(i, next_y)
      log_probs += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=next_y)
      return i + 1, next_c, next_h, actions, log_probs

    last_c = tf.zeros(
        [num_children, self.hparams.hidden_size],
        dtype=tf.float32)
    
    loop_vars = [tf.constant(0, dtype=tf.int32), last_c, last_h, actions,
        tf.zeros([num_children], dtype=tf.float32)]

    loop_outputs = tf.while_loop(condition, body, loop_vars)
    
    last_c = loop_outputs[-4]
    last_h = loop_outputs[-3]
    actions = loop_outputs[-2].stack()
    actions = tf.transpose(actions, [1, 0])
    log_probs = loop_outputs[-1]
    return actions, log_probs
  
  def encode_forward_random(self, random_logits):
    next_y = tf.multinomial(random_logits, 1, seed=self.hparams.seed)
    next_y = tf.to_int32(next_y)
    next_y = tf.reshape(next_y, [1]) #only generate one random action at a time
    return next_y

  def encode_forward(self, target_op, input_features, num_children, prev_c, prev_h, device_utilization, mode = "sample"):
    ph = self.placeholders

    with tf.variable_scope(self.hparams.name, reuse=tf.AUTO_REUSE):
      w_lstm = tf.get_variable("decoder_lstm")
      forget_bias = tf.get_variable("decoder_forget_bias")
      device_softmax = tf.get_variable("device_softmax")
      node_embedding = tf.get_variable("node_embedding")
      w_utilization = tf.get_variable("device_utilization")
    
    signal = tf.slice(input_features, [target_op, 0], [1, -1])
    signal = tf.reshape(signal, [self.hparams.feat_size])
    signal = tf.expand_dims(signal, 0)
    signal = tf.tile(signal, [num_children, 1])
    signal = tf.reshape(signal, [num_children, self.hparams.feat_size])
    
    if self.hparams.keep_prob is not None:
      signal = tf.nn.dropout(signal, self.hparams.keep_prob)
    
    feature_embedding = tf.matmul(signal, node_embedding)
    if self.hparams.consider_device_utilization:
      utilization_embedding = tf.matmul(device_utilization, w_utilization)
      feature_embedding = tf.concat([feature_embedding, utilization_embedding], axis = 1)
    if self.hparams.consider_neighbor_placement:
      node_sources_num = self.placeholders['node_sources_num']
      source_devices = self.placeholders['source_devices']
      source_weights = self.placeholders['source_weights']
      source_devices_embeddings = tf.cond(tf.equal(node_sources_num, 0), lambda: tf.zeros([num_children, self.hparams.hidden_size], dtype=tf.float32), lambda: self.aggregate_source_devices_forward(num_children, source_devices, source_weights, node_sources_num))
      feature_embedding = tf.concat([feature_embedding, source_devices_embeddings], axis=1)
    
    next_c, next_h = lstm(feature_embedding, prev_c, prev_h, w_lstm, forget_bias)
    logits = tf.matmul(next_h, device_softmax)
    
    if mode == "sample":
      next_y = tf.multinomial(logits, 1, seed=self.hparams.seed)
    elif mode == "greedy":
      next_y = tf.argmax(logits, 1)
    else:
      raise NotImplementedError
    
    next_y = tf.to_int32(next_y)
    next_y = tf.reshape(next_y, [num_children])
    return next_y, next_c, next_h

  def encode(self, input_features, num_ops, num_children, device_utilization, y):
    ph = self.placeholders
    sources = ph["batch_sources"]
    num_sources = ph["batch_num_sources"]
    source_weights = ph["batch_source_weights"]

    with tf.variable_scope(self.hparams.name, reuse=tf.AUTO_REUSE):
      w_lstm = tf.get_variable("decoder_lstm")
      forget_bias = tf.get_variable("decoder_forget_bias")
      device_softmax = tf.get_variable("device_softmax")
      node_embedding = tf.get_variable("node_embedding")
      w_utilization = tf.get_variable("device_utilization")

    actions = tensor_array_ops.TensorArray(
        tf.int32,
        size=num_ops,
        infer_shape=False,
        clear_after_read=False)
    
    def condition(i, *args):
      return tf.less(i, num_ops)

    def body(i, prev_c, prev_h, actions, log_probs):
      signal = tf.slice(input_features, [i, 0], [1, -1])
      signal = tf.reshape(signal, [self.hparams.feat_size])
      signal = tf.expand_dims(signal, 0)
      signal = tf.tile(signal, [num_children, 1])
      signal = tf.reshape(signal, [num_children, self.hparams.feat_size])
      
      if self.hparams.keep_prob is not None:
        signal = tf.nn.dropout(signal, self.hparams.keep_prob)
      feature_embedding = tf.matmul(signal, node_embedding)
      
      if self.hparams.consider_device_utilization:
        utilization = tf.slice(device_utilization, [i, 0, 0], [1, -1, -1])
        utilization = tf.reshape(utilization, [num_children, self.num_devices])
        utilization_embedding = tf.matmul(utilization, w_utilization)
        feature_embedding = tf.concat([feature_embedding, utilization_embedding], axis = 1)
      if self.hparams.consider_neighbor_placement:
        node_sources_num = tf.gather(num_sources, i)
        source_devices_embeddings = tf.cond(tf.equal(node_sources_num, 0), lambda: tf.zeros([num_children, self.hparams.hidden_size], dtype=tf.float32), lambda: self.aggregate_source_devices(i, actions, num_children, sources, node_sources_num, source_weights))
        feature_embedding = tf.concat([feature_embedding, source_devices_embeddings], axis = 1)
      next_c, next_h = lstm(feature_embedding, prev_c, prev_h, w_lstm, forget_bias)
      logits = tf.matmul(next_h, device_softmax)
      
      next_y = tf.slice(y, [0, i], [-1, 1])
      
      next_y = tf.to_int32(next_y)
      next_y = tf.reshape(next_y, [num_children])
      actions = actions.write(i, next_y)
      log_probs += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=next_y)
      return i + 1, next_c, next_h, actions, log_probs

    last_c = tf.zeros(
          [num_children, self.hparams.hidden_size],
          dtype=tf.float32)
    last_h = tf.zeros(
          [num_children, self.hparams.hidden_size],
          dtype=tf.float32)
    
    loop_vars = [tf.constant(0, dtype=tf.int32), last_c, last_h, actions,
        tf.zeros([num_children], dtype=tf.float32)]

    loop_outputs = tf.while_loop(condition, body, loop_vars)
    
    last_c = loop_outputs[-4]
    last_h = loop_outputs[-3]
    actions = loop_outputs[-2].stack()
    actions = tf.transpose(actions, [1, 0])
    log_probs = loop_outputs[-1]
    return actions, log_probs

  def optimize(self, fd):
    feed_dict = dict()

    feed_dict.update({self.placeholders['num_actions'] : fd['num_actions']})
    feed_dict.update({self.placeholders['actions'] : fd['actions']})
    feed_dict.update({self.placeholders['reward'] : fd['reward']})
    feed_dict.update({self.placeholders['batch'] : fd['batch']})
    feed_dict.update({self.placeholders['batch_sources'] : fd['batch_sources']})
    feed_dict.update({self.placeholders['batch_num_sources'] : fd['batch_num_sources']})
    feed_dict.update({self.placeholders['batch_size'] : fd['batch_size']})
    feed_dict.update({self.placeholders['graph_idx'] : fd['graph_idx']})
    feed_dict.update({self.placeholders['device_utilizations'] : fd['device_utilizations']})
    feed_dict.update({self.placeholders['batch_source_weights'] : fd['batch_source_weights']})
    
    controller_ops = self.ctr
    
    run_ops = [controller_ops["probs"], 
        controller_ops["loss"], controller_ops["lr"],
        controller_ops["grad_norm"], controller_ops["grad_norms"],
        controller_ops["train_op"]
    ]
    probs, loss, _, _, _, _ = self.sess.run(run_ops, feed_dict=feed_dict)

  def _get_train_ops(self,
                     loss,
                     tf_variables,
                     global_step,
                     grad_bound=1.25,
                     lr_init=1e-3,
                     lr_dec=0.9,
                     start_decay_step=10000,
                     decay_steps=100,
                     optimizer_type="adam"):
    def f1():
      return tf.constant(lr_init)

    def f2():
      return tf.train.exponential_decay(lr_init, lr_gstep, decay_steps, lr_dec, True)

    learning_rate = tf.cond(
        tf.less(global_step, start_decay_step),
        f1,
        f2,
        name="learning_rate")

    if optimizer_type == "adam":
      opt = tf.train.AdamOptimizer(learning_rate)
    elif optimizer_type == "sgd":
      opt = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = opt.compute_gradients(loss, tf_variables)
    grad_norm = tf.global_norm([g for g, v in grads_and_vars])
    all_grad_norms = {}
    clipped_grads = []
    clipped_rate = tf.maximum(grad_norm / grad_bound, 1.0)
    for g, v in grads_and_vars:
      if g is not None:
        if isinstance(g, tf.IndexedSlices):
          clipped = g.values / clipped_rate
          norm_square = tf.reduce_sum(clipped * clipped)
          clipped = tf.IndexedSlices(clipped, g.indices)
        else:
          clipped = g / clipped_rate
          norm_square = tf.reduce_sum(clipped * clipped)
        all_grad_norms[v.name] = tf.sqrt(norm_square)
        clipped_grads.append((clipped, v))

    train_op = opt.apply_gradients(clipped_grads, global_step)
    return train_op, learning_rate, grad_norm, all_grad_norms


def lstm(x, prev_c, prev_h, w_lstm, forget_bias):
  ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w_lstm)
  i, f, o, g = tf.split(ifog, 4, axis=1)
  i = tf.sigmoid(i)
  f = tf.sigmoid(f + forget_bias)
  o = tf.sigmoid(o)
  g = tf.tanh(g)
  next_c = i * g + f * prev_c
  next_h = o * tf.tanh(next_c)
  return next_c, next_h
