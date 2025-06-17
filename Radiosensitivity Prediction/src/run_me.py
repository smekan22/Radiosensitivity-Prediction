import os, sys
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from keras.activations import linear, relu, tanh, leaky_relu
from keras.losses import MeanSquaredError
from functools import partial
from sklearn.kernel_ridge import KernelRidge

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *
from architecture.callbacks_custom import step_decay, FixedEarlyStopping

# Download data if not done so already and set up run directory
wd = "Radiosensitivity Prediction"
download_dir = f"{wd}/data"
data_dir = f"{download_dir}/Cleveland"
run_dir = f"{wd}/runs"

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# Identify protein coding only genes and selected gene list based on P-Net paper
selected_genes = list(set(pd.read_csv(f"{download_dir}/hugo_genes.txt", sep="\t")["symbol"]))

# prepare config
config["data_dir"] = data_dir
config["run_dir"] = run_dir
config["run_id"] = "pnet"
config["views"] = [("gexpr", f"cleveland_gene_expression.csv", selected_genes, 0, lambda x : x, lambda x : x),
                   ("methylation", f"CCLE_Methylation_TSS1kb_20181022.csv", selected_genes, 0, lambda x : x, lambda x : x)]
config["view_alignment_method"] = "drop samples"
config["labels"] = [("cleveland_auc_only.csv", 0)]
config["tv_split_seed"] = 42
config["inner_kfolds"] = 5
config["outer_kfolds"] = 1
config["test_samples"] = 0.1
config["use_validation_on_test"] = False
config["val_metric"] = lambda x : r2_score(x["val_df"].ys, x["val_preds"])
config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"r2" : r2_score,
                                                                                    "explained_variance" : explained_variance_score,
                                                                                    "mse" : mean_squared_error,
                                                                                    "mae" : mean_absolute_error}, 
                                                                          "individual"),
                            plot_history]

n_hidden_layers = 5

step_decay_part = partial(
    step_decay,
    init_lr=0.001,
    drop=0.5,
    epochs_drop=25,
)

config["fitting_params"] = {
                                "epochs" : 200,
                                "batch" : 50,
                                "LRScheduler" : LearningRateScheduler(step_decay_part, verbose=0),
                                "early_stopping" : None,
                                "prediction_output" : "average",
                                "shuffle_samples" : True,
                                "class_weight" : None
                            }

gs_params = {"model_params" : {f"reg_{l}" : {
                            "pp_relations" : "architecture/Reactome/ReactomePathwaysRelation.txt",
                            "gp_relations" : "architecture/Reactome/ReactomePathways.gmt",
                            "n_hidden_layers" : n_hidden_layers,
                            "h_dropout" : [0.5] + [0.1] * n_hidden_layers,
                            "h_activation" : ["tanh"] * (n_hidden_layers + 1),
                            "o_activation" : ["linear"] * (n_hidden_layers + 1),
                            "h_reg" : [(L2, {"l2" : l})] * (n_hidden_layers + 1),
                            "o_reg" : [(L2, {"l2" : l})] * (n_hidden_layers + 1),
                            "h_kernel_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_kernel_constraints" : [None] * (n_hidden_layers + 1),
                            "h_bias_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_bias_constraints" : [None] * (n_hidden_layers + 1),
                            "batch_normal" : False,
                            "sparse" : True,
                            "dropout_testing" : False,
                            "loss" : ["MeanSquaredError"] * (n_hidden_layers + 1),
                            "loss_weights" : [2, 7, 20, 54, 148, 400],
                            "optimizer" : {"class_name" : "Adam", "config" : {"learning_rate" : 1e-3}}
                        } for l in [1, 0.1, 0.01, 0.001]}}

config["grid_search"] = construct_gs_params(gs_params)

# Run pnet crossvalidation
pipeline = TFPipeline(config)
pipeline.run_crossvalidation()

# Run dense crossvalidation
gs_params = {"model_params" : {f"reg_{l}" : {
                            "pp_relations" : "architecture/Reactome/ReactomePathwaysRelation.txt",
                            "gp_relations" : "architecture/Reactome/ReactomePathways.gmt",
                            "n_hidden_layers" : n_hidden_layers,
                            "h_dropout" : [0.5] + [0.1] * n_hidden_layers,
                            "h_activation" : ["tanh"] * (n_hidden_layers + 1),
                            "o_activation" : ["linear"] * (n_hidden_layers + 1),
                            "h_reg" : [(L2, {"l2" : l})] * (n_hidden_layers + 1),
                            "o_reg" : [(L2, {"l2" : l})] * (n_hidden_layers + 1),
                            "h_kernel_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_kernel_constraints" : [None] * (n_hidden_layers + 1),
                            "h_bias_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_bias_constraints" : [None] * (n_hidden_layers + 1),
                            "batch_normal" : False,
                            "sparse" : False,
                            "dropout_testing" : False,
                            "loss" : ["MeanSquaredError"] * (n_hidden_layers + 1),
                            "loss_weights" : [2, 7, 20, 54, 148, 400],
                            "optimizer" : {"class_name" : "Adam", "config" : {"learning_rate" : 1e-3}}
                        } for l in [1, 0.1, 0.01, 0.001]}}

config["grid_search"] = construct_gs_params(gs_params)
config["run_id"] = "dense"
pipeline = TFPipeline(config)
pipeline.run_crossvalidation()

# Run Kernel Regression
gs_params = {"model_params" : {f"degree_{d}_alpha_{a}" : {"kernel" : "poly", "degree" : d, "alpha" : a}
                               for d in [1, 2, 3] for a in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]}}
config["model"] = KernelRidge
config["run_id"] = "krr"
config["task"] = "regression"
config["results_processors"] = config["results_processors"][:-1]
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
"""
# Compile results
def compile_results(tag, gridsearch):
    results = []
    for i, fold in enumerate([x for x in os.listdir(f"{run_dir}/{tag}") if os.path.isdir(f"{run_dir}/{tag}/{x}")]):
        for cv in [x for x in os.listdir(f"{run_dir}/{tag}/{fold}") if os.path.isdir(f"{run_dir}/{tag}/{fold}/{x}")]:
            with open(f"{run_dir}/{tag}/{fold}/{cv}/config.txt") as f:
                run_config = json.loads(f.read())
            result = pd.read_csv(f"{run_dir}/{tag}/{fold}/{cv}/summary_results.csv")
            result.columns = ["split"] + list(result.columns[1:])
            result["test_fold"] = i
            for k,v in gridsearch.items():
                result[k] = run_config[v]
            results.append(result)
    results = pd.concat(results)
    val_means = results.loc[results["split"] == "val"].groupby(list(gridsearch.keys()) + ["test_fold"])["auc_r2"].mean().reset_index()
    top = val_means.iloc[val_means.groupby("test_fold")["auc_r2"].idxmax(), ]
    filt = None
    for k,v in gridsearch.items():
        if filt is None:
            filt = results[k] == top[k].iat[0]
        else:
            filt = (results[k] == top[k].iat[0]) & filt
    results = results.loc[filt]
    aggresults = results.groupby("split")[["auc_r2", "auc_explained_variance", "auc_mse", "auc_mae"]].agg(["mean", "std"])
    hyperparams = "_".join([top[k].iat[0] for k in gridsearch.keys()])
    aggresults.to_csv(f"{wd}/{tag}_{hyperparams}_results.csv")
    return results.loc[results["split"] == "test"]

dense_results = compile_results("dense", {"reg" : "model_params_choice", "es" : "fitting_params_choice"})
pnet_results = compile_results("pnet", {"reg" : "model_params_choice", "es" : "fitting_params_choice"})
krr_results = compile_results("krr", {"hyper" : "model_params_choice"})

metrics = ["auc_r2", "auc_explained_variance", "auc_mse", "auc_mae"]
pvd = [ttest_ind(pnet_results[x], dense_results[x]).pvalue for x in metrics]
pvk = [ttest_ind(pnet_results[x], krr_results[x]).pvalue for x in metrics]
svd = [ttest_ind(krr_results[x], dense_results[x]).pvalue for x in metrics]
sigresults = pd.DataFrame((pvd, pvk, svd), columns=metrics, index=["pnet_v_dense", "pnet_v_krr", "krr_v_dense"])
sigresults.to_csv(f"{wd}/significance_tests.csv")
"""

