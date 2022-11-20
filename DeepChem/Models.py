import deepchem as dc 
from deepchem.models import GraphConvModel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import tempfile

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets

def model_builder_svc(model_dir):
  sklearn_model = SVC(C=1.0, class_weight="balanced", probability=True)
  return dc.models.SklearnModel(sklearn_model, model_dir)


model_dir = tempfile.mkdtemp()
model_svc = dc.models.SingletaskToMultitask(tox21_tasks, model_builder_svc, model_dir)

# Fit trained SVC model
model_svc.fit(train_dataset)


def model_builder_rf(model_dir):
  sklearn_model = RandomForestClassifier(class_weight="balanced")
  return dc.models.SklearnModel(sklearn_model, model_dir)


model_dir = tempfile.mkdtemp()
model_rf = dc.models.SingletaskToMultitask(tox21_tasks, model_builder_rf, model_dir)

# Fit trained RF model
model_rf.fit(train_dataset)


from deepchem.models import GraphConvModel
np.random.seed(0)

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

print("Evaluating Support Vector Classifier")
train_scores_svc = model_svc.evaluate(train_dataset, [metric_rocauc, metric_recall, metric_f1, metric_accuracy], transformers)
print("Training ROC-AUC Scores: %f" % (train_scores_svc["mean-roc_auc_score"]))
print("Training Recall Scores: %f" % (train_scores_svc["recall_score"]))
print("Training F1 Scores: %f" % (train_scores_svc["f1_score"]))
print("Training Accuracy Scores: %f" % (train_scores_svc["accuracy_score"]))

valid_scores_svc = model_svc.evaluate(valid_dataset, [metric_rocauc, metric_recall, metric_f1, metric_accuracy], transformers)
print("Validation ROC-AUC Scores: %f" % (valid_scores_svc["mean-roc_auc_score"]))
print("Validation Recall Scores: %f" % (valid_scores_svc["recall_score"]))
print("Validation F1 Scores: %f" % (valid_scores_svc["f1_score"]))
print("Validation Accuracy Scores: %f" % (valid_scores_svc["accuracy_score"]))


test_scores_svc = model_svc.evaluate(test_dataset, [metric_rocauc, metric_recall, metric_f1, metric_accuracy], transformers)
print("Test ROC-AUC Scores: %f" % (test_scores_svc["mean-roc_auc_score"]))
print("Test Recall Scores: %f" % (test_scores_svc["recall_score"]))
print("Test F1 Scores: %f" % (test_scores_svc["f1_score"]))
print("Test Accuracy Scores: %f" % (test_scores_svc["accuracy_score"]))

print("Evaluating Random Forest")
train_scores_rf = model_rf.evaluate(train_dataset, [metric_rocauc, metric_recall, metric_f1, metric_accuracy], transformers)
print("Training ROC-AUC Scores: %f" % (train_scores_rf["mean-roc_auc_score"]))
print("Training Recall Scores: %f" % (train_scores_rf["recall_score"]))
print("Training F1 Scores: %f" % (train_scores_rf["f1_score"]))
print("Test Accuracy Scores: %f" % (train_scores_rf["accuracy_score"]))


valid_scores_rf = model_rf.evaluate(valid_dataset, [metric_rocauc, metric_recall, metric_f1, metric_accuracy], transformers)
print("Validation ROC-AUC Scores: %f" % (valid_scores_rf["mean-roc_auc_score"]))
print("Validation Recall Scores: %f" % (valid_scores_rf["recall_score"]))
print("Validation F1 Scores: %f" % (valid_scores_rf["f1_score"]))
print("Test Accuracy Scores: %f" % (valid_scores_rf["accuracy_score"]))


test_scores_rf = model_rf.evaluate(test_dataset, [metric_rocauc, metric_recall, metric_f1, metric_accuracy], transformers)
print("Test ROC-AUC Scores: %f" % (test_scores_rf["mean-roc_auc_score"]))
print("Test Recall Scores: %f" % (test_scores_rf["recall_score"]))
print("Test F1 Scores: %f" % (test_scores_rf["f1_score"]))
print("Test Accuracy Scores: %f" % (test_scores_rf["accuracy_score"]))

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
