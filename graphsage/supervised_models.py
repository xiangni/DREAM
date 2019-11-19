import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator, MaxPoolingGraphAggregator

class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, placeholders, features,
            layer_infos, batch_size = 32, concat=True, aggregator_type="mean", 
            model_size="small", name = None, **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)
      
        self.graph_aggregator_cls = MaxPoolingGraphAggregator
        
        # get info from placeholders...
        self.model_size = model_size
        
        #features: shape (num_graphs, num_nodes, 2)
        #get the features related to this graph
        #self.features = features #tf.Variable(tf.constant(features[self.g_id], dtype=tf.float32), trainable=False)
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)

        self.concat = concat
        print(self.features.shape)
        self.dims = [int(self.features.shape[2])] #last dimention
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        print(self.dims)
        self.placeholders = placeholders

        self.layer_infos = layer_infos

        self.env_batch_size = batch_size

        #self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build_aggregators()

        self.build()


    def build_aggregators(self):
        self.aggregators = []
        self.num_layers = len(self.layer_infos)
        for layer in range(self.num_layers):
            dim_mult = 2 if self.concat and (layer != 0) else 1
            # aggregator at current layer
            if layer == self.num_layers - 1:
                aggregator = self.aggregator_cls(dim_mult*self.dims[layer],
                self.dims[layer+1], act=lambda x : x, dropout=self.placeholders['dropout'], 
                        concat=self.concat, model_size=self.model_size)
                self.output_dim = self.dims[layer+1] * dim_mult

            else:
                aggregator = self.aggregator_cls(dim_mult*self.dims[layer], self.dims[layer+1],
                        dropout=self.placeholders['dropout'], 
                        concat=self.concat, model_size=self.model_size)
            self.aggregators.append(aggregator)

        self.graph_aggregator = self.graph_aggregator_cls(self.output_dim*2,
                                    self.output_dim*2, act=lambda x : x, 
                                    dropout=self.placeholders['dropout'],  model_size=self.model_size)
    
    def build(self):
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
          
        g_id = self.placeholders['graph_idx']

        samples1, support_sizes1 = self.sample(g_id, self.placeholders['batch'], self.placeholders['batch_size'], True)
        
        features = tf.slice(self.features, [g_id, 0, 0], [1, -1, -1])
        features = tf.reshape(features, [self.features.shape[1], self.features.shape[2]])
        outputs1 = self.aggregate(samples1, [features], num_samples,
              support_sizes1, self.placeholders['batch_size'],
              concat=self.concat, model_size=self.model_size)

        outputs1 = tf.nn.l2_normalize(outputs1, 1)
        
        samples2, support_sizes2 = self.sample(g_id, self.placeholders['batch'], self.placeholders['batch_size'], False)
        
        features = tf.slice(self.features, [g_id, 0, 0], [1, -1, -1])
        features = tf.reshape(features, [self.features.shape[1], self.features.shape[2]])
        #features = tf.gather(self.features, g_id)
        #num_nodes = features.shape[0]
        #num_feats = features.shape[1]
        #features = tf.reshape(features, [num_nodes, num_feats])
        outputs2 = self.aggregate(samples2, [features], num_samples,
              support_sizes2, self.placeholders['batch_size'],
              concat=self.concat, model_size=self.model_size)

        outputs2 = tf.nn.l2_normalize(outputs2, 1)

        self.node_preds = tf.concat([outputs1, outputs2], axis = 1)
        self.graph_preds = self.graph_aggregator(self.node_preds)


    def get_node_preds(self):
        return self.node_preds

    def get_graph_preds(self):   
        return self.graph_preds
    
    def sample(self, g_id, inputs, batch_size, ins):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(self.layer_infos)):
            t = len(self.layer_infos) - k - 1
            support_size *= self.layer_infos[t].num_samples
            sampler = self.layer_infos[t].neigh_sampler
            node = sampler((g_id, samples[k], self.layer_infos[t].num_samples, ins))
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes


    def aggregate(self, samples, input_features, num_samples, support_sizes, batch_size,
            name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """


        # length: number of layers + 1
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        for layer in range(len(num_samples)):
            aggregator = self.aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*self.dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]
