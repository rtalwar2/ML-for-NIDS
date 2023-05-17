# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



datasets={}
for dirname, _, filenames in os.walk('/project_ghent/raman/netflow_datasets'):
    for filename in filenames:
        if filename.endswith("parquet") and  "NF-UNSW-NB15-V2" not in filename:
            datasets[filename.replace(".parquet","")]=pd.read_parquet(os.path.join(dirname, filename))
            
            
for k,v in datasets.items():
    print(v.Attack.unique())
    
    
for ds,data in datasets.items():
    ddos=data[data['Attack'].str.contains("ddos",case=False)]
    ddos.Attack="DDoS"#uniform name
    benign=data[data['Attack'] == 'Benign']
    
    size_ddos=ddos.shape[0]
    size_benign= benign.shape[0]
    
    if size_ddos>size_benign:
        #downsample attack
        ddos= ddos.sample(size_benign)
    else:
        #downsample benign
        benign = benign.sample(size_ddos)
    
    data = pd.concat(objs=[ddos, benign])
    
    # random forest classifier works internally with float32 so values too big will give errors
    data = data[data['SRC_TO_DST_SECOND_BYTES'] <= np.finfo(np.float32).max]
    data = data[data['DST_TO_SRC_SECOND_BYTES'] <= np.finfo(np.float32).max]
    data=data.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT']) # dropping metadata

    
    datasets[ds]=data
    
for k,v in datasets.items():
    print(v.Attack.value_counts())
    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
for k,v in datasets.items():
    x=v[v.columns[:-2]]
    y=v[v.columns[-2]]
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    stratify=y, 
                                                    test_size=0.75)
    scaler = StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    y_train.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)
    datasets[k]={"X_train":X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test}
    print(f"{k}: {y_train.shape[0]} {y_test.shape[0]}")

dataset_names = datasets.keys()

from sklearn.neighbors import KNeighborsClassifier
from os import cpu_count
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
dataset_names = datasets.keys()
prepared_result_rows = [] 
for dataset_train in dataset_names:
    for dataset_test in dataset_names:
            accuracies = []
            precisions = []
            recalls = []
            for i in range(10):
                clf=KNeighborsClassifier(n_jobs=-1)
                clf.fit(X=datasets[dataset_train]["X_train"], y=datasets[dataset_train]["y_train"])
                outputs = clf.predict(X=datasets[dataset_test]["X_test"])

                accuracies.append(accuracy_score(y_true=datasets[dataset_test]["y_test"], y_pred=outputs))
                precisions.append(precision_score(y_true=datasets[dataset_test]["y_test"], y_pred=outputs, zero_division=0))
                recalls.append(recall_score(y_true=datasets[dataset_test]["y_test"], y_pred=outputs, zero_division=0))
            prepared_result_row = [
                            dataset_train,
                            dataset_test,
                            round(np.mean(accuracies), 3),
                            round(np.std(accuracies), 3),
                            round(np.mean(precisions), 3),
                            round(np.std(precisions), 3),
                            round(np.mean(recalls), 3),
                            round(np.std(recalls), 3),            
            ]
            print(prepared_result_row)
            prepared_result_rows.append(prepared_result_row)
            
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[2] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(len(dataset_names),-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DDoS accuracy")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/accuracy_heatmap_ddos_knn.png")

fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[4] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(len(dataset_names),-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DDoS precision")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/precision_heatmap_ddos_knn.png")

fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[6] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(len(dataset_names),-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DDoS recall")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/recall_heatmap_ddos_knn.png")
