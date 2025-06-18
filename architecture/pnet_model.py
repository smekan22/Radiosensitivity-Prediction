
import re, os
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import tensorflow as tf
import logging
from keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)
from keras import Input, Model

from architecture.layers_custom import *
from architecture.callbacks_custom import TQDMCallback

class TFModel:
    """
    Class wrapper around a TensorFlow Model
    """
    def __init__(self, run_id, model, model_params, fitting_params):
        """
        Constructor for the Model class
        
        args:
            run_id (str) : tag to identify this specific model run
            model (function that returns a Model object) : Function used to build the desired model
            model_params (dict) : dictionary containing the parameters required for calling the model function
            fitting_params (dict) : dictionary containing parameters for fitting the model
        """
        self.set_params(run_id, model, model_params, fitting_params)

    def set_params(self, run_id, model, model_params, fitting_params):
        """
        Support function to enable flexible resetting of config
        
        args:
            run_id (str) : tag to identify this specific model run
            model (function that returns a Model object) : Function used to build the desired model
            model_params (dict) : dictionary containing the parameters required for calling the model function
            fitting_params (dict) : dictionary containing parameters for fitting the model
        """
        self.model = model
        self.run_id = run_id
        self.fitting_params = fitting_params
        self.model_params = model_params
    
    def get_callbacks(self):
        """
        Adds callbacks to facilitate adaptive learning rate, early stopping, and monitoring fitting progress
        as specified in fitting_params config
        """
        callbacks = []
        # Monitor progress of fitting
        callbacks.append(TQDMCallback(self.fitting_params["epochs"]))
        # Add early stopping if defined
        if self.fitting_params["early_stopping"] is not None:
            callbacks.append(self.fitting_params["early_stopping"])
        # Add adaptive learning rate if defined
        if self.fitting_params["LRScheduler"] is not None:
            callbacks.append(self.fitting_params["LRScheduler"])
        return callbacks

    def fit(self, train_df, val_df, seed):
        """
        Function to perform fitting of the model

        args:
            train_df (ConcatMultiviewDataset) : Training data object
            val_df (ConcatMultiviewDataset) : Validation data object
            seed (int) : seed to make fitting reproducible

        returns:
            the model object, training history
        """
        # Clear previous runs
        if hasattr(self, "predictor"):
            del self.predictor
        if hasattr(self, "feature_names"):
            del self.feature_names
        
        # Set reproducibility seeds and enable determinism
        tf.keras.backend.clear_session()
        tf.compat.v1.random.set_random_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

        # Build model
        logging.info("building pnet")
        self.model_params["data"] = train_df
        model = self.model(**self.model_params)
        if type(model) is tuple:
            self.predictor, self.feature_names = model
        else:
            self.predictor = model

        # Set callbacks
        callbacks = self.get_callbacks()

        # Fit the model
        logging.info("start fitting")
        if len(val_df) > 0:
            history = self.predictor.fit(
                train_df.xs,
                [train_df.ys] * (self.model_params["n_hidden_layers"] + 1),
                validation_data=(val_df.xs, [val_df.ys] * (self.model_params["n_hidden_layers"] + 1)),
                epochs=self.fitting_params["epochs"],
                batch_size=self.fitting_params["batch"],
                class_weight=self.fitting_params["class_weight"] if "class_weight" in self.fitting_params.keys() else None,
                verbose=0,
                callbacks=callbacks,
                shuffle=self.fitting_params["shuffle_samples"]
            )
        else:
            history = self.predictor.fit(
                train_df.xs,
                [train_df.ys] * (self.model_params["n_hidden_layers"] + 1),
                epochs=self.fitting_params["epochs"],
                batch_size=self.fitting_params["batch"],
                class_weight=self.fitting_params["class_weight"] if "class_weight" in self.fitting_params.keys() else None,
                verbose=0,
                callbacks=callbacks,
                shuffle=self.fitting_params["shuffle_samples"]
            )
        
        return self, history
    
    def get_prediction_score(self, X):
        prediction_scores = self.predictor.predict(X)
        if type(prediction_scores) == list:
            if len(prediction_scores) > 1:
                if self.fitting_params["prediction_output"] == "average":
                    prediction_scores = np.mean(np.array(prediction_scores), axis=0)
                    print(f"Shape of prediction scores : {prediction_scores.shape}")
                else:
                    prediction_scores = prediction_scores[-1]

        return np.array(prediction_scores)
    
    def predict(self, X):
        return self.get_prediction_score(X)
    
    def predict_proba(self, X_test):
        prediction_scores = self.get_prediction_score(X_test)
        if type(X_test) is list:
            n_samples = X_test[0].shape[0]
        else:
            n_samples = X_test.shape[0]
        ret = np.ones((n_samples, 2))
        ret[:, 0] = 1.0 - prediction_scores.ravel()
        ret[:, 1] = prediction_scores.ravel()
        return ret
    
    def save_model(self, filename):
        model_json = self.predictor.to_json()
        json_file_name = filename.replace(".h5", ".json")
        with open(json_file_name, "w") as json_file:
            json_file.write(model_json)
        self.predictor.save_weights(filename)

    def load_model(self, filename):
        ret = self.model(**self.model_params)
        if type(ret) == tuple:
            self.predictor, self.feature_names = ret
        else:
            self.predictor = ret

        self.predictor.load_weights(filename)

        return self

def compile_pnet(pp_relations, gp_relations, n_hidden_layers, optimizer, loss, loss_weights, data, 
                 h_activation, o_activation, h_reg, o_reg, h_dropout, sparse, batch_normal, h_kernel_initializer,
                 h_kernel_constraints, h_bias_initializer, h_bias_constraints, dropout_testing):
    """
    Compiles P-Net model specifying the optimizer, loss function, and loss weights on top of building
    the P-Net architecture

    args
        pp_relations (str) : Path to reactome pathway relations file
        gp_relations (str) : Path to reactome pathway gmt file
        n_hidden_layers (int) : Depth of p-net hierarchy to construct
        optimizer (Keras Optimizer) : Optimizer to be used for fitting P-Net
        loss (str or loss function) : Loss to be applied to each outcome layer of P-Net
        loss_weights (list[float]) : Weight to be applied to the loss of each outcome layer of P-Net
        data (ConcatMultiViewDataset) : Object containing information on the data inputs for the model
        h_activation (list[keras activation]) : List of activation functions to use per hidden layer
        o_activation (list[keras activation]) : List of activation functions to use per outcome layer
        h_reg (list[tuple]) : List of tuples containing keras regularizer and its dict of parameters for each hidden layer
        o_reg (list[tuple]) : List of tuples containing keras regularizer and its dict of parameters for each outcome layer
        h_dropout (list[float]) : List of dropout rates to use on each hidden layer
        sparse (bool) : Whether to treat connections between hidden layers as sparse of densely connected
        batch_normal (bool) : Whether to perform batch normalization
        h_kernel_initializer (list[keras initializer]) : Initialization method to use for each hidden layer's parameters
        h_kernel_constraints (list[keras constraints]) : Constraint to be used for each hidden layer's parameters
        h_bias_initializer (list[keras initializer]) : Initialization method to use for each hidden layer's bias
        h_bias_constraints (list[keras constraints]) : Constraint to be used for each hidden layer's bias
        dropout_testing (bool) : Whether to apply dropout outside of training

    """

    # Extract information on the data to be used with p-net
    x = data.xs
    y = data.ys
    info = data.ids
    features = data.get_features()
    genes = sorted(list(set(data.get_alignment_ids())))

    logging.info(
        "x shape {} , y shape {} info {} genes {}".format(
            x.shape, y.shape, len(info), len(features)
        )
    )

    # Specify input dimension for P-Net
    inputs = Input(shape=(x.shape[1],), dtype="float32", name="inputs")

    # Build P-Net structure
    reactome = PNetArchitectureGenerator()
    netx = reactome.get_reactome_networkx(pp_relations)
    maps = reactome.get_layers(netx, n_hidden_layers, gp_relations, data.get_alignment_ids())
    maps = get_layer_maps(genes, maps, False)
    _, decision_outcomes, feature_names = build_pnet(inputs, data, maps[:-1], h_activation, o_activation,
             h_reg, o_reg, h_dropout, sparse, batch_normal, h_kernel_initializer,
             h_kernel_constraints, h_bias_initializer, h_bias_constraints, dropout_testing)

    # Compile P-Net with the opimizer and loss function
    logging.info("Compiling...")
    model = Model(inputs=[inputs], outputs=decision_outcomes)
    model.compile(
        optimizer=optimizer,
        loss=[tf.keras.losses.get(l) for l in loss],
        metrics=[],
        loss_weights=loss_weights,
    )
    logging.info("done compiling")
    print(model.summary())
    logging.info("# of trainable params of the model is %s" % model.count_params())

    return model, feature_names

def build_pnet(inputs, data, maps, h_activation, o_activation,
             h_reg, o_reg, h_dropout, sparse, batch_normal, h_kernel_initializer,
             h_kernel_constraints, h_bias_initializer, h_bias_constraints, dropout_testing):
    """
    Function which builds the P-Net model using Keras library

    args:
        inputs (TensorFlow tensor) : Shape of inputs
        data (ConcatMultiViewDataset) : Object containing information on the data inputs for the model
        map (list[dict]) : List containing the relationship between layers for the P-Net model
        h_activation (list[keras activation]) : List of activation functions to use per hidden layer
        o_activation (list[keras activation]) : List of activation functions to use per outcome layer
        h_reg (list[tuple]) : List of tuples containing keras regularizer and its dict of parameters for each hidden layer
        o_reg (list[tuple]) : List of tuples containing keras regularizer and its dict of parameters for each outcome layer
        h_dropout (list[float]) : List of dropout rates to use on each hidden layer
        sparse (bool) : Whether to treat connections between hidden layers as sparse of densely connected
        batch_normal (bool) : Whether to perform batch normalization
        h_kernel_initializer (list[keras initializer]) : Initialization method to use for each hidden layer's parameters
        h_kernel_constraints (list[keras constraints]) : Constraint to be used for each hidden layer's parameters
        h_bias_initializer (list[keras initializer]) : Initialization method to use for each hidden layer's bias
        h_bias_constraints (list[keras constraints]) : Constraint to be used for each hidden layer's bias
        dropout_testing (bool) : Whether to apply dropout outside of training
    
    returns:
        outcome, decision_outcomes, feature_names : output of the hidden layers, output of the decision layers, feature labels 
                                                    for each hidden layer
    """
    # Keep track of number of features and number of genes
    feature_names = {}
    # Start constructing first layer from input features to set of genes
    gene_set = np.sort(np.unique(np.array(data.get_alignment_ids())))
    feature_gene_map = Diagonal(len(gene_set), tf.keras.activations.get(h_activation[0]),h_bias_initializer[0] is not None,
                                 tf.keras.initializers.get(h_kernel_initializer[0]),  tf.keras.initializers.get(h_bias_initializer[0]), 
                                 h_reg[0][0](**h_reg[0][1]), None, tf.keras.constraints.get(h_kernel_constraints[0]), 
                                 tf.keras.constraints.get(h_bias_constraints[0]), name="h0")
    feature_names["inputs"] = data.get_features()
    # Create first decision outcome layer
    out_0 = Dense(data.ys.shape[1], tf.keras.activations.get(o_activation[0]), kernel_regularizer=o_reg[0][0](**o_reg[0][1]), name="o_linear_0")
    # Create first dropout layer
    dropout_0 = Dropout(h_dropout[0], name="dropout_0")
    # Apply layers
    outcome = feature_gene_map(inputs)
    decision_outcomes = [BatchNormalization()(out_0(outcome))] if batch_normal else [out_0(outcome)]
    outcome = dropout_0(outcome, training=dropout_testing)

    # Construct P-Net hidden layer hierarchy
    inp_features = gene_set
    for i, map in enumerate(maps):
        out_features = np.array(sorted(map.keys()))
        logging.info("================================")
        logging.info(f'PROCEEDING TO LAYER " {i}')
        logging.info("================================")
        logging.info("Print the information on the PNET layers")
        logging.info("no. inputs, no. outputs {} {} ".format(len(inp_features), len(out_features)))
        logging.info("layer {}, dropout  {} w_reg {}".format(i, h_dropout[i+1], h_reg[i+1]))
        layer_name = "h{}".format(i + 1)
        if sparse:
            # Construct sparse connection matrix from list of dicts
            hidden_layer = SparseTF(len(out_features), map, None, tf.keras.initializers.get(h_kernel_initializer[i+1]),
                                h_reg[i+1][0](**h_reg[i+1][1]), tf.keras.activations.get(h_activation[i+1]), 
                                h_bias_initializer[i+1] is not None, tf.keras.initializers.get(h_bias_initializer[i+1]), None, 
                                tf.keras.constraints.get(h_kernel_constraints[i+1]), tf.keras.constraints.get(h_bias_constraints[i+1]), name=layer_name)
        else:
            hidden_layer = Dense(len(out_features), tf.keras.activations.get(h_activation[i+1]), h_bias_initializer[i+1] is not None,
                                 tf.keras.initializers.get(h_kernel_initializer[i+1]), tf.keras.initializers.get(h_bias_initializer[i+1]), 
                                 h_reg[i+1][0](**h_reg[i+1][1]), None, None, tf.keras.constraints.get(h_kernel_constraints[i+1]), 
                                 tf.keras.constraints.get(h_bias_constraints[i+1]), name=layer_name)
        # Apply hidden layer
        outcome = hidden_layer(outcome)
        # Get decision
        out_n = Dense(data.ys.shape[1], tf.keras.activations.get(o_activation[i+1]), 
                      kernel_regularizer=o_reg[i+1][0](**o_reg[i+1][1]), name="o_linear_{}".format(i+1))
        decision_outcome = BatchNormalization()(out_n(outcome)) if batch_normal else out_n(outcome)
        decision_outcomes.append(decision_outcome)
        # Apply dropout
        dropout_n = Dropout(h_dropout[i+1])
        outcome = dropout_n(outcome, training=dropout_testing)
        # Save feature names
        feature_names["h{}".format(i)] = inp_features
        inp_features = out_features

    # Save last layer of feature names
    feature_names["h{}".format(i+1)] = inp_features

    return outcome, decision_outcomes, feature_names

class PNetArchitectureGenerator:
    """
    Wrapper class to contain the code for constructing PNET reactome tree from the original
    PNET paper.
    """
    def get_reactome_networkx(self, pp_relations : str):
        hierarchy = pd.read_csv(pp_relations, sep="\t")
        hierarchy.columns = ["child", "parent"]
        # filter hierarchy to have human pathways only
        human_hierarchy = hierarchy[hierarchy["child"].str.contains("HSA")]
        net = nx.from_pandas_edgelist(
            human_hierarchy, "child", "parent", create_using=nx.DiGraph()
        )
        net.name = "reactome"

        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = "root"
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net
    
    def get_terminals(self, net):
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]
        return terminal_nodes

    def get_roots(self):
        roots = self.get_nodes_at_level(distance=1)
        return roots
    
    def get_nodes_at_level(self, net, distance):
        # get all nodes within distance around the query node
        nodes = set(nx.ego_graph(net, "root", radius=distance))

        # remove nodes that are not **at** the specified distance but closer
        if distance >= 1.0:
            nodes -= set(nx.ego_graph(net, "root", radius=distance - 1))

        return list(nodes)
    
    def add_edges(self, G, node, n_levels):
        edges = []
        source = node
        for l in range(n_levels):
            target = node + "_copy" + str(l + 1)
            edge = (source, target)
            source = target
            edges.append(edge)

        G.add_edges_from(edges)
        return G
    
    def complete_network(self, G, n_leveles=4):
        sub_graph = nx.ego_graph(G, "root", radius=n_leveles)
        terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
        distances = [
            len(nx.shortest_path(G, source="root", target=node)) for node in terminal_nodes
        ]
        for node in terminal_nodes:
            distance = len(nx.shortest_path(sub_graph, source="root", target=node))
            if distance <= n_leveles:
                diff = n_leveles - distance + 1
                sub_graph = self.add_edges(sub_graph, node, diff)

        return sub_graph
    
    def info(self, net):
        return nx.info(net)

    def get_tree(self, net):

        # convert to tree
        G = nx.bfs_tree(net, "root")

        return G

    def get_completed_network(self, net, n_levels):
        G = self.complete_network(net, n_leveles=n_levels)
        return G

    def get_completed_tree(self, net, n_levels):
        G = self.get_tree(net)
        G = self.complete_network(G, n_leveles=n_levels)
        return G
    
    def get_layers_from_net(self, net, n_levels):
        layers = []
        for i in range(n_levels):
            nodes = self.get_nodes_at_level(net, i)
            dict = {}
            for n in nodes:
                n_name = re.sub("_copy.*", "", n)
                next = net.successors(n)
                dict[n_name] = [re.sub("_copy.*", "", nex) for nex in next]
            layers.append(dict)
        return layers

    def get_layers(self, net, n_levels, gp_relations, alignment_ids):
        net = self.get_completed_network(net, n_levels)
        layers = self.get_layers_from_net(net, n_levels)

        # get the last layer (genes level)
        terminal_nodes = [
            n for n, d in net.out_degree() if d == 0
        ]  # set of terminal pathways
        # we need to find genes belonging to these pathways
        # And only include those which are present in the given features
        genes_df = self.load_gmt(gp_relations, genes_col=3, pathway_col=1)
        genes_df = genes_df.loc[genes_df["gene"].isin(alignment_ids)]

        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub("_copy.*", "", p)
            genes = genes_df[genes_df["group"] == pathway_name]["gene"].unique()
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = genes

        layers.append(dict)
        return layers
    
    def load_gmt(self, filename, genes_col=1, pathway_col=0):
        data_dict_list = []
        with open(filename) as gmt:
            data_list = gmt.readlines()
            for row in data_list:
                genes = row.strip().split('\t')
                genes = [re.sub('_copy.*', '', g) for g in genes]
                genes = [re.sub('\\n.*', '', g) for g in genes]
                for gene in genes[genes_col:]:
                    pathway = genes[pathway_col]
                    dict = {'group': pathway, 'gene': gene}
                    data_dict_list.append(dict)

        df = pd.DataFrame(data_dict_list)
        return df

def get_map_from_layer(layer_dict):
    """
    :param layer_dict: dictionary of connections (e.g {'pathway1': ['g1', 'g2', 'g3']}
    """
    pathways = list(layer_dict.keys())

    genes = list(itertools.chain.from_iterable(list(layer_dict.values())))
    genes = list(np.unique(genes))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in list(layer_dict.items()):
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1

    df = pd.DataFrame(mat, index=pathways, columns=genes)

    return df.T


def get_layer_maps(genes, reactome_layers, add_unk_genes):
    """
    :param genes: list of genes
    :param n_levels: number of layers
    :param direction: direction of the graph
    :param add_unk_genes: {True, False}
    :return: list of maps (dataframes) for each layer
    """
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):

        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)

        filtered_map = filter_df.merge(
            mapp, right_index=True, left_index=True, how="left"
        )
        print(filtered_map)
        print("test end")
        # UNK, add a node for genes without known reactome annotation
        if add_unk_genes:

            filtered_map["UNK"] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, "UNK"] = 1

        # Handling missing values, using pandas database to fill NaN values with 0
        filtered_map = filtered_map.fillna(0)

        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        logging.info("layer %s , # of edges  %s", i, filtered_map.sum().sum())
        maps.append(filtered_map.sort_index().sort_index(axis=1))  # list of maps (dataframes) for each layer
    return maps