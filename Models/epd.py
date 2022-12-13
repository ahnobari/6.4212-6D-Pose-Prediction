import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset, download_url

import numpy as np
import tqdm

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_geometric.nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
class MLP_block(nn.Module):
    def __init__(self, in_features: int, num_hidden_layers: int, hidden_size: int, output_size: int, use_layer_norm: bool):
        super().__init__()
        
        layers = []
        # Add hidden layers
        layers.append(
                nn.Linear(in_features, hidden_size, bias=True))
        layers.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers-1):
            layers.append(
                nn.Linear(hidden_size, hidden_size, bias=True))
            layers.append(torch.nn.ReLU())

        # Add output layer
        layers.append(nn.Linear(hidden_size, output_size, bias=True))

        if use_layer_norm:
            layers.append(nn.LayerNorm(output_size))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class Encoder(nn.Module):
    """Encodes an input graph into a latent graph with the same structure but
    different node and edge attributes.

    New embeddings are computed independently, i.e. edge and node-wise:
        latent_edge_embeddings = edge_encode_func(edge_attributes)
        latent_node_embeddings = node_encode_func(node_attributes)"""

    def __init__(self, node_encode_func, edge_encode_func):
        """Initializes a Encoder object.

        Args:
            node_encode_func: Callable function used to encode node attributes.
              Ideally it should be a keras.layers.Layer object with trainable
              parameters.
            edge_encode_func: Callable function used to encode edge attributes.
              Ideally it should be a keras.layers.Layer object with trainable
              parameters."""

        super().__init__()

        self.node_encode_func = node_encode_func
        self.edge_encode_func = edge_encode_func

    def forward(self, x):
        """Forward pass of the encoder.

        Args:
            inputs: list containing [X, E]. X is a 2D array of shape
              (n_nodes, n_node_attributes) with the node features and E is a 2D
              array of shape (n_edges, n_edge_attributes) with the edge
              attributes.

        Returns:
            X: 2D array of shape (n_nodes, n_node_attributes) with the new node
              embeddings.
            E: 2D array of shape (n_edges, n_edge_attributes) with the new edge
              embeddings."""
        #pylint: disable=invalid-name

        X, E = x

        # Compute node embeddings
        X = self.node_encode_func(X)

        # Compute edge embeddings
        E = self.edge_encode_func(E)

        return X, E
    
class Decoder(nn.Module):
    """Decodes latent attributes of a graph's nodes into output attributes.
    Output of each node is computed independently:
        output[i] = node_decode_func(latent_x[i])
    """

    def __init__(self, node_decode_func):
        """Initializes a Decoder object.

        Args:
            node_decode_func: Callable function used to decode node attributes.
              Ideally it should be a keras.layers.Layer object with trainable
              parameters.
        """
        super().__init__()

        self.node_decode_func = node_decode_func

    def forward(self, x):
        """Forward pass of the Decoder.

        Args:
            inputs: 2D array of shape (n_nodes, n_latent_node_attributes) with
                the latent attributes to be decoded.

        Returns:
            outputs: 2D array of shape (n_nodes, n_output_node_attributes) with
              the output attributes of every node.
        """

        # Compute outputs from latent node embeddings
        outputs = self.node_decode_func(x)

        return outputs
    
class Processor(MessagePassing):
    """Updates node and edge attributes of a graph, performing message passing.
    Graph Network computational block, as defined by [1], that performs message
    passing on a graph. Implementation is similar to Interaction Network
    described in [1] (see appendix):

        1. Update edge embeddings by considering the edge and the its nodes:
            new_e[i,j] = edge_update_func(concat(e[i,j], x[i], x[j]]))
            The new edge embeddings will also be the messages sent to its nodes.

        2. For every node, aggregate the new edge embeddings (messages) it is
           connected to:
            agg_e[i] = aggregate(new_e[i,:])

        3. Update node embeddings by considering the node and the aggregated
           messages:
           new_x[i] = node_update_func(concat(agg_e[i], x[i]))

    [1] P. W. Battaglia et al., ‘Relational inductive biases, deep learning,
    and graph networks’. arXiv, Oct. 17, 2018. Accessed: Jun. 21, 2022.
    [Online]. Available: http://arxiv.org/abs/1806.01261
"""

    def __init__(self,node_update_func,edge_update_func,aggregation_func):
        """Initializes a Processor object.

        Args:
            node_update_func: Callable function used to update node attributes.
              Ideally it should be a keras.layers.Layer object with trainable
              parameters.
            edge_update_func: Callable function used to update edge attributes.
              Ideally it should be a keras.layers.Layer object with trainable
              parameters.
            aggregation_func: A permutation-invariant callable function to
              aggregate messages of each node. Should be a function in the
              module spektral.layers.ops.
        """
        super().__init__()

        self.edge_update_func = edge_update_func
        self.aggregation_func = aggregation_func
        self.node_update_func = node_update_func
        
    def forward(self, x, edge_index, edge_features):

        return self.propagate(edge_index, x=x, edge_attr=edge_features)
    
    def message(self, x_i, x_j, edge_attr):
        """Computes messages to be transmitted from nodes to their neighbours.
        Messages are computed as a function of the node attributes and the edge
        attributes that connect them. Edge attributes are updated with the
        computed messages.
            message[i,j] = edge_update_func(concat(e[i,j], x[i], x[j]]))
            new_e[i,j] = message[i,j]

        Args:
            x: 2D array of shape (n_nodes, n_node_features) with node features.
            e: 2D array of shape (n_edges, n_edge_features) with edge features.

        Returns:
            new_edge_attributes: 2D array of shape (n_edges, n_edge_features)
              with the updated edge attributes. These new edge attributes are
              also the messages passed between the corresponding nodes.
        """
        
        # Concatenate edge attributes with corresponding node attributes
        new_edge_attributes = torch.cat([edge_attr, x_i, x_j],dim=1)
        
        # Update edge attributes
        new_edge_attributes = self.edge_update_func(new_edge_attributes)

        return new_edge_attributes

    def aggregate(self, inputs,index):
        """Aggregates the messages sent to each node. If a node has N
        neighbours, it receives N messages, which are then aggregated by a
        permutation invariant function (eg. sum, mean, average, etc.).

        Args:
            messages: 2D array of shape (n_edges, n_edge_features)
              with the updated edge attributes. These new edge attributes are
              also the messages passed between the corresponding nodes.

        Returns:
            aggregated_edge_attributes: 2D array of shape
              (n_nodes, n_edge_features) with the aggregated messages of each
              node.
            messages: The messages themselves provided as input, which also
              represent the new edge attributes.
        """
        new_edge_attributes = inputs
        # For each node, aggregate new attributes (messages) of its edges
        aggregated_edge_attributes = self.aggregation_func(new_edge_attributes,index)

        return aggregated_edge_attributes, new_edge_attributes

    def update(self, inputs, x):
        """Update the node attributes as a function of the aggregated messages
        (new edge attributes) and the old node attributes.
            new_x[i] = node_update_func(concat(agg_e[i], x[i]))

        Args:
            embeddings: list containing [aggregated_edge_attributes, messages]
              returned by the aggregate function.
            x: 2D array of shape (n_nodes, n_node_features) with the old node
              features.

        Returns:
            new_node_attributes: 2D array of shape (n_nodes, n_node_features)
              with the new updated node features.
            new_edge_attributes: 2D array of shape (n_edges, n_edge_features)
              with the new edge features. These also represent the messages
              computed by the message function.
        """
        aggregated_edge_attributes, new_edge_attributes = inputs

#         # Concatenate node attributes with the aggregated messages
        new_node_attributes = torch.cat([aggregated_edge_attributes, x],dim=1)

        # Update node embeddings
        new_node_attributes = self.node_update_func(new_node_attributes)

        return new_node_attributes, new_edge_attributes
    
class EncoderProcessorDecoder(nn.Module):
    """Learnable model combining the three steps: Encoder, Processor, Decoder.
    This model takes as input a graph, encodes it into a latent representation,
    processes it with message passing steps and finally decodes into node
    output attributes. Implementation is similar to the one described in [1].

    Let X be an array of shape (n_nodes, n_node_attributes) with the node
    attributes, E be an array of shape (n_edges, n_edge_attributes) with the
    edge attributes and A be an array of shape (n_nodes, n_nodes) representing
    an adjacency matrix. Then, this model performs the following steps:

        1. Encode edge and node attributes as latent embeddings:

            X_latent, E_latent = Encoder(X, E)

            Latent embeddings are computed by learnable MLPs, one for nodes and
            another for edges.

        2. Process the latent graph with message passing steps and update
           embeddings:

            X_latent, E_latent = Processor(X_latent, A, E_latent)

            Latent embeddings are updated by learnable MLPs, one for nodes and
            another for edges.

        3. Decode the final latent node embeddings into output attributes:

            X_out = Decoder(X_latent)

            Output node attributes are decoded by a learnable MLP.

    [1] A. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, and
    P. W. Battaglia, 'Learning to Simulate Complex Physics with Graph Networks',
    arXiv:2002.09405 [physics, stat], Sep. 2020, Accessed: Mar. 02, 2022.
    [Online]. Available: http://arxiv.org/abs/2002.09405
"""

    def __init__(self,
                 node_in_features: int,
                 edge_in_features: int,
                 latent_size: int,
                 mlp_hidden_size: int,
                 mlp_num_hidden_layers: int,
                 use_layer_norm: bool,
                 num_message_passing_steps: int,
                 output_size: int,
                 aggregation_func=torch_geometric.nn.aggr.SumAggregation()):
        """Initializes a EncoderProcessorDecoder object.

        Args:
            latent_size: Size of the node and edge latent representations.
            mlp_hidden_size: Hidden layer size for all MLPs.
            mlp_num_hidden_layers: Number of hidden layers in all MLPs.
            use_layer_norm: Boolean indicating whether to use LayerNorm after
              the outputs of the MLPs output.
            num_message_passing_steps: Number of message passing steps in the
              processor.
            output_size: Size of the decoded output node representations.
            aggregation_func: A permutation-invariant callable function to
              aggregate messages of each node. Use functions in the module
              spektral.layers.ops.
            normalize_input_features: Whether to normalize input node and edge
              features to zero mean and unit variance.
            noise_std: Standard deviation of Gaussian noise added to velocity
              features when training. If `None`, no noise is added.
        """
        super().__init__()

        self.num_message_passing_steps = num_message_passing_steps

        self.encoder =Encoder(
            node_encode_func=MLP_block(node_in_features,mlp_num_hidden_layers, mlp_hidden_size,
                                        latent_size, use_layer_norm),
            edge_encode_func=MLP_block(edge_in_features,mlp_num_hidden_layers, mlp_hidden_size,
                                        latent_size, use_layer_norm)
        )

        self.processor = []

        # Add 'num_message_passing_steps' Processor blocks to the model and the
        # corresponding residual connections for node and edge features.
        for _ in range(self.num_message_passing_steps):
            self.processor.append(
                Processor(
                    node_update_func=MLP_block(2*latent_size, mlp_num_hidden_layers,
                                                mlp_hidden_size, latent_size,
                                                use_layer_norm),
                    edge_update_func=MLP_block(3*latent_size, mlp_num_hidden_layers,
                                                mlp_hidden_size, latent_size,
                                                use_layer_norm),
                    aggregation_func=aggregation_func))
        
        self.processor = nn.ModuleList(self.processor)
        
        self.decoder = Decoder(node_decode_func=MLP_block(latent_size, mlp_num_hidden_layers, mlp_hidden_size, output_size, False))
        
    def forward(self, inputs):
        """Forward pass of the EncoderProcessorDecoder.

        Args:
            inputs: list containing [X, A, E].
                    - X is a 2D array of shape (n_nodes, n_node_attributes)
                      with the node features.
                    - A is a 2D array of shape (n_nodes, n_nodes) representing
                      an adjacency matrix. It should be in SparseTensor type.
                      Use spektral.utils.sp_matrix_to_sp_tensor() to cast.
                    - E is a 2D array of shape (n_edges, n_edge_attributes)
                      with the edge attributes.

            output: 2D array of shape (n_nodes, output_size) with the decoded
              attributes of each node.
        """
        #pylint: disable=invalid-name

        X, A, E = inputs  # when a batch of graphs is given

        X, E = self.encoder((X, E))

        for i in range(self.num_message_passing_steps):
            X_new, E_new = self.processor[i](X, A, E)

            X = X + X_new
            E = E + E_new

        output = self.decoder(X)

        return output
