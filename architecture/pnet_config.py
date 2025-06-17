from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.data_utils import ConcatMultiViewDataset
from architecture.pipeline import IdentityProcessor
from architecture.pnet_model import compile_pnet
from architecture.callbacks_custom import step_decay_part
from architecture.evaluation import collate_grid_search

n_hidden_layers = 5

config = {
    "dataloader" : ConcatMultiViewDataset,
    "feature_selector" : IdentityProcessor(),
    "feature_preprocessor" : IdentityProcessor(),
    "data_augmentor" : lambda x : x,
    "rng_seed" : 42,
    "tt_split_seed" : 42,
    "model" : compile_pnet,
    "model_params" : {
        "pp_relations" : "architecture/Reactome/ReactomePathwaysRelation.txt",
        "gp_relations" : "architecture/Reactome/ReactomePathways.gmt",
        "n_hidden_layers" : n_hidden_layers,
        "h_dropout" : [0.5] + [0.1] * n_hidden_layers,
        "h_activation" : ["tanh"] * (n_hidden_layers + 1),
        "o_activation" : ["sigmoid"] * (n_hidden_layers + 1),
        "h_reg" : [(L2, {"l2" : 1e-3})] * (n_hidden_layers + 1),
        "o_reg" : [(L2, {"l2" : 1e-2})] * (n_hidden_layers + 1),
        "h_kernel_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
        "h_kernel_constraints" : [None] * (n_hidden_layers + 1),
        "h_bias_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
        "h_bias_constraints" : [None] * (n_hidden_layers + 1),
        "batch_normal" : False,
        "sparse" : True,
        "dropout_testing" : False,
        "loss" : [{"class_name" : "BinaryCrossentropy", "config" : {"from_logits" : False}}] * (n_hidden_layers + 1),
        "loss_weights" : [2, 7, 20, 54, 148, 400],
        "optimizer" : {"class_name" : "Adam", "config" : {"learning_rate" : 1e-3}}
    },
    "fitting_params" : {
        "epochs" : 300,
        "batch" : 50,
        "LRScheduler" : LearningRateScheduler(step_decay_part, verbose=0),
        "early_stopping" : None,
        "prediction_output" : "average",
        "shuffle_samples" : True,
        "class_weight" : [[0.75, 1.5]] * (n_hidden_layers + 1)
    },
    "grid_search" : [],
    "val_metric" : lambda x : x,
    "use_validation_on_test" : True,
    "results_processors" : [],
    "fold_collators" : [],
    "grid_search_collators" : [collate_grid_search],
    "drop_labels" : True
}