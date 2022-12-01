import deepchem as dc 
from deepchem.models import GraphConvModel
import numpy as np 


# Load Tox21 dataset
tox21_tasks_gcn, tox21_datasets_gcn, transformers_gcn = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset_gcn, valid_dataset_gcn, test_dataset_gcn = tox21_datasets_gcn

model_gcn = GraphConvModel(
    len(tox21_tasks_gcn), batch_size=64, mode='classification', random_seed=0)
model_gcn.fit(train_dataset_gcn, nb_epoch=2000,  deterministic=True)


# Evaluating
metric_rocauc = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
metric_recall = dc.metrics.Metric(dc.metrics.recall_score)
metric_f1 = dc.metrics.Metric(dc.metrics.f1_score)
metric_accuracy = dc.metrics.Metric(dc.metrics.accuracy_score)


print("Evaluating GCN model")
train_scores_gcn = model_gcn.evaluate(train_dataset_gcn, [metric_rocauc, metric_recall, metric_f1, metric_accuracy], transformers_gcn)
print("Training ROC-AUC Scores: %f" % (train_scores_gcn["mean-roc_auc_score"]))
print("Training Recall Scores: %f" % (train_scores_gcn["recall_score"]))
print("Training F1 Scores: %f" % (train_scores_gcn["f1_score"]))
print("Test Accuracy Scores: %f" % (train_scores_gcn["accuracy_score"]))



valid_scores_gcn = model_gcn.evaluate(valid_dataset_gcn, [metric_rocauc, metric_recall, metric_f1, metric_accuracy], transformers_gcn)
print("Validation ROC-AUC Scores: %f" % (valid_scores_gcn["mean-roc_auc_score"]))
print("Validation Recall Scores: %f" % (valid_scores_gcn["recall_score"]))
print("Validation F1 Scores: %f" % (valid_scores_gcn["f1_score"]))
print("Test Accuracy Scores: %f" % (valid_scores_gcn["accuracy_score"]))


test_scores_gcn = model_gcn.evaluate(test_dataset_gcn, [metric_rocauc, metric_recall, metric_f1, metric_accuracy], transformers_gcn)
print("Test ROC-AUC Scores: %f" % (test_scores_gcn["mean-roc_auc_score"]))
print("Test Recall Scores: %f" % (test_scores_gcn["recall_score"]))
print("Test F1 Scores: %f" % (test_scores_gcn["f1_score"]))
print("Test Accuracy Scores: %f" % (test_scores_gcn["accuracy_score"]))


# XAI
