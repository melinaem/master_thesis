import deepchem as dc 
from sklearn.svm import SVC
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
