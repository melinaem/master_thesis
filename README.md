# master_thesis



This repository is a part of the master's thesis ''. In this thesis, we investigate aspects of machine learning models in the computational toxicology domain. Toxicity evaluation of compounds is one of the main challenges during drug developments and is the second reason, after lack of efficacy, for failure in preclinical and clinical trials. To reduce attrition at later stages of the drug development process as well as reduce cost and time, toxicity analyses should be employed at early stages of the drug development pipeline and the methods utlized should be reliable and able to accurately predict toxicity of compounds. To investigate how certain machine learnign models perform, we will use the data set 'Tox21 - The Toxicology in the 21st Century', which comprises samples that represent chemical compounds and can be used to predict toxicity. This repository contains a CSV-file of the data set, located at ..., but other methods of accessing the data set, such as through the Python library 'DeepChem' and PyTorch Geometric, is used to develop the machine learning models. The illustration below presents the study design of the master's thesis.



![workflow](https://user-images.githubusercontent.com/62059573/202899885-7ac6b7b3-6791-423f-a312-0f9bd9b7a3f3.png)


## Getting started

Python language version 3.9.15 is used. 

```
$ pip3 install -r requirements.txt
$ jupyter notebook
```

## Content

The `Data` folder contains a CSV-file of the Tox21 data set and the `DeepChem` folder contains a python file with implementation of machine learning algorithms developed using the DeepChem library. These algorithms include a SVC, a RF, and a GCN. The `PyTorch Geometric` folder contains a python file with implementation of a GCN using the library PyTorch Geometric. The `Results` folder contains the runs from the code with results and performances of all models. The Notebook folder contains a Jupyter `Notebook` of the code, while the `XAI` folder contains python files that include code for applying the XAI methods LIME, SHAP, and GNNExplainer on various data sets.
    
