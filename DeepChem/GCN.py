import deepchem as dc 
from deepchem.models import GraphConvModel
import numpy as np 
import time
from datetime import timedelta

time_gcn = time.time()

np.random.seed(0)

# Load Tox21 dataset
tox21_tasks_gcn, tox21_datasets_gcn, transformers_gcn = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset_gcn, valid_dataset_gcn, test_dataset_gcn = tox21_datasets_gcn

model_gcn = GraphConvModel(
    len(tox21_tasks_gcn), batch_size=64, mode='classification', random_seed=0)

model_gcn.fit(train_dataset_gcn, nb_epoch=200,  deterministic=True)

t_gcn = timedelta(seconds= (time.time() - time_gcn))
print("--- Execution time: %s  ---" % (t_gcn))


# For exploring training loss
time_gcn_loss = time.time()

np.random.seed(123)

model_gcn = GraphConvModel(
    len(tox21_tasks_gcn), batch_size=64, mode='classification', random_seed=0)

losses = []
for epoch in range(1, 200, 10):
    loss = model_gcn.fit(train_dataset_gcn, nb_epoch=epoch,  deterministic=True) 
    losses.append(loss)
    if epoch % 10 == 0:
      print(f"Epoch {epoch} | Train Loss {loss}")


t_gcn_loss = timedelta(seconds= (time.time() - time_gcn_loss))
print("--- Execution time: %s  ---" % (t_gcn_loss))

# Plot training lossses
import matplotlib.pyplot as plt

#losses_plot_train = [losses[i] for i in range(0, len(losses), 10)]
default_x_ticks = range(1, 200, 10)
plt.plot(default_x_ticks, losses, 'g')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()


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


