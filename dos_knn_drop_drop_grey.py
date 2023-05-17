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
        if filename.endswith("parquet"):
            datasets[filename.replace(".parquet","")]=pd.read_parquet(os.path.join(dirname, filename))
            
            
for k,v in datasets.items():
    print(v.Attack.unique())
    
    
for ds,data in datasets.items():
    ddos=data[data['Attack'].str.contains("ddos",case=False)]
    dos=data[data['Attack'].str.contains("dos",case=False)]
    dos.drop(index=ddos.index,inplace=True)
    dos.Attack="DoS"#uniform name
    benign=data[data['Attack'] == 'Benign']
    
    size_dos=dos.shape[0]
    size_benign= benign.shape[0]
    
    if size_dos>size_benign:
        #downsample attack
        dos= dos.sample(size_benign)
    else:
        #downsample benign
        benign = benign.sample(size_dos)
    
    data = pd.concat(objs=[dos, benign])
    
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
    X_train=pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test=pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    y_train.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)
    datasets[k]={"X_train":X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test}
    print(f"{k}: {y_train.shape[0]} {y_test.shape[0]}")
    
    
contaminants={}
contaminants["NF-CSE-CIC-IDS2018-V2"]=["MIN_TTL", "MAX_TTL","TCP_WIN_MAX_OUT","NUM_PKTS_128_TO_256_BYTES","LONGEST_FLOW_PKT", 'MAX_IP_PKT_LEN',"SERVER_TCP_FLAGS","SHORTEST_FLOW_PKT","MIN_IP_PKT_LEN"]
contaminants["NF-CSE-CIC-IDS2018-V2_grey"]=[
    "L7_PROTO",
    "TCP_WIN_MAX_IN",
    "FLOW_DURATION_MILLISECONDS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "DURATION_OUT",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "OUT_BYTES",
    "OUT_PKTS",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_256_TO_512_BYTES"
]

contaminants["NF-BoT-IoT-V2"]=["TCP_WIN_MAX_IN"]
contaminants["NF-BoT-IoT-V2_grey"]=['LONGEST_FLOW_PKT', 'MAX_IP_PKT_LEN', 'CLIENT_TCP_FLAGS', 'DST_TO_SRC_SECOND_BYTES', 'OUT_BYTES', 'OUT_PKTS', 'SHORTEST_FLOW_PKT', 'DURATION_OUT']

contaminants["NF-ToN-IoT-V2"]=["TCP_WIN_MAX_IN"]
contaminants["NF-ToN-IoT-V2_grey"]=["LONGEST_FLOW_PKT", "MAX_IP_PKT_LEN", "SRC_TO_DST_AVG_THROUGHPUT", "IN_BYTES"]

contaminants["NF-UNSW-NB15-V2"]=["MIN_TTL", "MAX_TTL", "MIN_IP_PKT_LEN", "TCP_WIN_MAX_OUT", "TCP_WIN_MAX_IN", "PROTOCOL"]
contaminants["NF-UNSW-NB15-V2_grey"]=["DST_TO_SRC_SECOND_BYTES", "OUT_BYTES", "OUT_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS", "NUM_PKTS_1024_TO_1514_BYTES", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES", "IN_PKTS"]
    
    
    
    
dataset_names = datasets.keys()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from os import cpu_count
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
def inter_dataset():
    prepared_result_rows = []
    for dataset_train in dataset_names:
        for dataset_test in dataset_names:
                accuracies = []
                precisions = []
                recalls = []
                for i in range(10):
                    clf=KNeighborsClassifier(n_jobs=-1)
                    clf.fit(X=datasets[dataset_train]["X_train"], y=datasets[dataset_train]["y_train"])
                    outputs = clf.predict(X=datasets[dataset_test]["X_test"][datasets[dataset_train]["X_train"].columns])

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
    return prepared_result_rows
            
prepared_result_rows=inter_dataset()

fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[2] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(4,-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DoS accuracy")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/accuracy_heatmap_dos_knn.png")


fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[4] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(4,-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DoS precision")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/precision_heatmap_dos_knn.png")



fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[6] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(4,-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DoS recall")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/recall_heatmap_dos_knn.png")


print("drop contaminants")
for k,v in datasets.items():
    print(datasets[k]["X_train"].shape)
    datasets[k]["X_train"].drop(columns=contaminants[k],inplace=True)

prepared_result_rows=inter_dataset()

fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[2] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(4,-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DoS accuracy dropped")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/accuracy_heatmap_dos_knn_dropped.png")


fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[4] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(4,-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DoS precision dropped")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/precision_heatmap_dos_knn_dropped.png")



fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[6] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(4,-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DoS recall dropped")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/recall_heatmap_dos_knn_dropped.png")

print("also drop grey zone contaminants")
for k,v in datasets.items():
    print(datasets[k]["X_train"].shape)
    datasets[k]["X_train"].drop(columns=contaminants[k+"_grey"],inplace=True)
    print(datasets[k]["X_train"].shape)
    
prepared_result_rows=inter_dataset()

fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[2] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(4,-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DoS accuracy extra dropped")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/accuracy_heatmap_dos_knn_extra_dropped.png")


fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[4] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(4,-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DoS precision extra dropped")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/precision_heatmap_dos_knn_extra_dropped.png")



fig, ax = plt.subplots(figsize=(15, 15))

stacked_array_norm = np.stack([row[6] for row in prepared_result_rows])
stacked_array_norm=stacked_array_norm.reshape(4,-1)
# create heatmap with annotations
sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

# set title and labels
ax.set_title("inter-dataset generalization DoS recall extra dropped")
ax.set_xlabel('test')
ax.set_ylabel('train')

# set x and y ticklabels to categorical labels
ax.set_xticklabels(dataset_names)
ax.set_yticklabels(dataset_names, rotation=0, ha='right')
plt.subplots_adjust(left=0.2)
plt.savefig(f"/project_ghent/raman/images/recall_heatmap_dos_knn_extra_dropped.png")
