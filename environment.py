import heapq
import sys
import numpy as np
from collections import deque

class Sample(object):
  
  def __init__(self, action, throughput, device_utilization, is_policy = False):
    self.action = action
    self.throughput = throughput
    p = ''.join(str(i) for i in action)
    self.action_str = p
    self.rank = 0
    self.device_utilization = device_utilization
    self.is_policy = is_policy
  
  def set_rank(self, rank):
    self.rank = rank

THRE = 0.8
class Environment(object):
  
  def __init__(self, graph_idx, batch_size, max_throughput, queue_leangth = 200):
    self.graph_idx = graph_idx
    self.exploration_buf = set()
    self.min_throughput = 0.0
    self.max_throughput = max_throughput
    self.replay_buffer = []
    self.policy_queue = deque()
    self.policy_queue_length = queue_leangth
    self.num_policy_samples = 0
    self.exploration_done = False
    self.best_rank = 0.0
    self.perf_map = {}

  def set_exploration_done(self):
    self.exploration_done = True
    
  def has(self, action):
    p = ''.join(str(i) for i in action)
    if p not in self.exploration_buf:
      return False
    return True

  def if_exist(self, action_str):
    if action_str in self.perf_map:
      throughput = self.perf_map[action_str]
      return throughput
    else:
      return -1

  def save(self, s, on_policy=False, build_replay = True):
    self.perf_map[s.action_str] = s.throughput
    if build_replay and (s.action_str not in self.exploration_buf):
      self.exploration_buf.add(s.action_str)
      self.replay_buffer.append(s)
    
    s.rank = s.throughput/self.max_throughput
    
    if s.rank > self.best_rank:
      self.best_rank = s.rank

    if on_policy:
      if self.num_policy_samples > self.policy_queue_length:
        self.policy_queue.popleft()
        self.num_policy_samples -= 1
      
      self.policy_queue.append(s)
      self.num_policy_samples += 1

  def save_test(self, throughput, action_str):
    self.perf_map[action_str] = throughput

  def calculate_baseline(self, epoch, num_replay_samples):
    if self.num_policy_samples == 0:
      return 0.0
    if epoch < 1:
      policy_samples = list(self.policy_queue)
      base = np.mean([s.rank for s in policy_samples])
    else:
      num_selected = min(2*num_replay_samples-1, len(self.replay_buffer))
      samples = heapq.nlargest(max(num_selected, 2), self.replay_buffer, key = lambda s: s.rank)
      base = np.mean([s.rank for s in samples])
    return base
  
  def sample(self, num_samples):
    policy_samples = list(self.policy_queue)
    start_index = len(policy_samples) - num_samples
    selected_samples = policy_samples[start_index:]
    return selected_samples
    
  def replay(self, num_samples, greedy = True):
    num_selected = min(num_samples, len(self.replay_buffer))
    
    if greedy:
      samples = heapq.nlargest(num_selected, self.replay_buffer, key = lambda s: s.rank)
      samples = np.random.choice(samples, num_selected, replace=False)
    else:
      samples = np.random.choice(self.replay_buffer, num_selected, replace=False)
    
    #filter the rank must be above 0.5
    samples = [x for x in samples if x.rank >= THRE]
    
    #select at least one top performer
    if len(samples) == 0:
      num_selected = 1
      samples = heapq.nlargest(num_selected, self.replay_buffer, key = lambda s: s.rank)
      samples = [x for x in samples]
    
    for s in samples:
      print("replay action: "+s.action_str + " rank: "+str(s.rank))
    print("max throughput {}".format(self.max_throughput))
    
    return samples
  
  def hard_problem(self):
    print("best rank {}".format(self.best_rank))
    if self.best_rank < 0.99:
      return True
    else:
      return False
  
