import deepchem as dc 
from sklearn.svm import SVC
import numpy as np 
import tempfile
import time 
from datetime import timedelta

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets


time_svc = time.time()

np.random.seed(2)

def model_builder_svc(model_dir):
  sklearn_model = SVC(C=1.0, class_weight="balanced", probability=True)
  return dc.models.SklearnModel(sklearn_model, model_dir)


model_dir = tempfile.mkdtemp()
model_svc = dc.models.SingletaskToMultitask(tox21_tasks, model_builder_svc, model_dir)

# Fit trained model
model_svc.fit(train_dataset) 

t_svc = timedelta(seconds= (time.time() - time_svc))
print("--- Execution time: %s  ---" % (t_svc))


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


# Apply XAI methods LIME to the SVC
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
    
# LIME for SVC model 
time_lime = time.time()

model_fn_svc = eval_model(model_svc)

# We want to investigate toxic compounds
actives_svc = []
n = len(np.where(test_dataset.y[:,0]==1)[0])

for i in range(n):
  actives_svc.append(np.where(test_dataset.y[:,0]==1)[0][i])

exps_svc = []
for active_id in actives_svc:
  exps_svc.append(explainer.explain_instance(test_dataset.X[active_id], model_fn_svc, num_features=5, top_labels=1))

t_lime = timedelta(seconds= (time.time() - time_lime))
print("--- Execution time: %s  ---" % (t_lime))


for i in np.where(test_dataset.y[:,0]==1)[0]: 
  print('Compound nr.: ', i, "is a toxic compound")

# Show what fragments the model believes contributed towards predicting toxic/non-toxic
for i in range(len(exps_svc)):
  exps_svc[i].show_in_notebook(show_table=True, show_all=False)
  
  
active_id = np.where(test_dataset.y[:,0]==1)[0][12] 

def fp_mol(mol, fp_length=1024):
    """
    returns: dict of <int:list of string>
        dictionary mapping fingerprint index
        to list of smile string that activated that fingerprint
    """
    d = {}
    feat = dc.feat.CircularFingerprint(sparse=True, smiles=True, size=1024)
    retval = feat._featurize(mol)
    for k, v in retval.items():
        index = k % 1024
        if index not in d:
            d[index] = set()
        d[index].add(v['smiles'])
    return d
# What fragments activated what fingerprints in our active molecule?
my_fp = fp_mol(Chem.MolFromSmiles(test_dataset.ids[active_id]))

# We can calculate which fragments activate all fingerprint
# indexes throughout our entire training set
all_train_fps = {}
X = train_dataset.X
ids = train_dataset.ids
for i in range(len(X)):
    d = fp_mol(Chem.MolFromSmiles(ids[i]))
    for k, v in d.items():
        if k not in all_train_fps:
            all_train_fps[k] = set()
        all_train_fps[k].update(v)
