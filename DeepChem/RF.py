import deepchem as dc 
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import tempfile

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets


def model_builder_rf(model_dir):
  sklearn_model = RandomForestClassifier(class_weight="balanced")
  return dc.models.SklearnModel(sklearn_model, model_dir)


model_dir = tempfile.mkdtemp()
model_rf = dc.models.SingletaskToMultitask(tox21_tasks, model_builder_rf, model_dir)

# Fit trained RF model
model_rf.fit(train_dataset)

# Evaluating
metric_rocauc = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
metric_recall = dc.metrics.Metric(dc.metrics.recall_score)
metric_f1 = dc.metrics.Metric(dc.metrics.f1_score)
metric_accuracy = dc.metrics.Metric(dc.metrics.accuracy_score)


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


# Apply XAI methods LIME to the RF
import lime
from lime.lime_tabular import LimeTabularExplainer 
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole

feature_names = ["fp_%s"  % x for x in range(1024)]
explainer = LimeTabularExplainer(train_dataset.X, 
                                              feature_names=feature_names, 
                                              categorical_features=feature_names,
                                              class_names=['not toxic', 'toxic'], 
                                              discretize_continuous=True)

def eval_model(my_model):
    def eval_closure(x):
        ds = dc.data.NumpyDataset(x, n_tasks=12)
        predictions = my_model.predict(ds)[:,0]
        return predictions
    return eval_closure
    
model_fn_rf = eval_model(model_rf)

# We want to investigate toxic compounds
actives_rf = []
n = len(np.where(test_dataset.y[:,0]==1)[0]) # number of toxic compounds

for i in range(n):
  actives_rf.append(np.where(test_dataset.y[:,0]==1)[0][i])

exps_rf = []
for active_id in actives_rf:
  exps_rf.append(explainer.explain_instance(test_dataset.X[active_id], model_fn_rf, num_features=5, top_labels=1))
  
  
for i in np.where(test_dataset.y[:,0]==1): #test og fiks denne
  print("Compound id" i)
  

# Show what fragments the model believes contributed towards predicting toxic/non-toxic
for i in range(len(exps_rf)):
  exps_rf[i].show_in_notebook(show_table=True, show_all=False)
