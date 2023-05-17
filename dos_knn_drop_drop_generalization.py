import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

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
    

correlations={}
correlations["NF-ToN-IoT-V2"]=[['OUT_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES'],
 ['TCP_FLAGS', 'SERVER_TCP_FLAGS'],
 ['DURATION_IN', 'DURATION_OUT'],
 ['MIN_TTL', 'MAX_TTL'],
 ['LONGEST_FLOW_PKT', 'MAX_IP_PKT_LEN'],
 ['RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS'],
 ['RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS'],
 ['ICMP_TYPE', 'ICMP_IPV4_TYPE']]

correlations["NF-BoT-IoT-V2"]=[['PROTOCOL', 'L7_PROTO'],
 ['IN_BYTES',
  'IN_PKTS',
  'NUM_PKTS_512_TO_1024_BYTES',
  'NUM_PKTS_1024_TO_1514_BYTES'],
 ['OUT_BYTES', 'OUT_PKTS'],
 ['TCP_FLAGS', 'SERVER_TCP_FLAGS', 'MIN_IP_PKT_LEN'],
 ['MIN_TTL', 'MAX_TTL'],
 ['LONGEST_FLOW_PKT', 'MAX_IP_PKT_LEN'],
 ['RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS'],
 ['RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS'],
 ['ICMP_TYPE', 'ICMP_IPV4_TYPE']]

correlations["NF-UNSW-NB15-V2"]=[['IN_BYTES', 'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS'],
 ['OUT_BYTES',
  'OUT_PKTS',
  'RETRANSMITTED_OUT_BYTES',
  'RETRANSMITTED_OUT_PKTS',
  'NUM_PKTS_1024_TO_1514_BYTES'],
 ['TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS'],
 ['MIN_TTL', 'MAX_TTL'],
 ['LONGEST_FLOW_PKT', 'MAX_IP_PKT_LEN'],
 ['ICMP_TYPE', 'ICMP_IPV4_TYPE']]

correlations["NF-CSE-CIC-IDS2018-V2"]=[['IN_BYTES', 'IN_PKTS', 'DURATION_IN', 'NUM_PKTS_UP_TO_128_BYTES'],
 ['OUT_BYTES', 'OUT_PKTS', 'NUM_PKTS_1024_TO_1514_BYTES'],
 ['TCP_FLAGS', 'CLIENT_TCP_FLAGS'],
 ['MIN_TTL', 'MAX_TTL'],
 ['LONGEST_FLOW_PKT', 'MAX_IP_PKT_LEN'],
 ['SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES'],
 ['RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS'],
 ['ICMP_TYPE', 'ICMP_IPV4_TYPE']]
    
generalizaion_contaminants=[
"MIN_TTL",
"MIN_IP_PKT_LEN",
"SHORTEST_FLOW_PKT",
"TCP_FLAGS",
"TCP_WIN_MAX_OUT",
"TCP_WIN_MAX_IN",
"DST_TO_SRC_SECOND_BYTES",
"OUT_BYTES",
"DST_TO_SRC_AVG_THROUGHPUT",
"NUM_PKTS_UP_TO_128_BYTES",
"IN_PKTS",
"LONGEST_FLOW_PKT",
"SRC_TO_DST_SECOND_BYTES",
"IN_BYTES",
"SRC_TO_DST_AVG_THROUGHPUT",
"ICMP_TYPE & ICMP_IPV4_TYPE",
"L7_PROTO",
"PROTOCOL",
"CLIENT_TCP_FLAGS",
"FLOW_DURATION_MILLISECONDS",
"DURATION_OUT",
"NUM_PKTS_128_TO_256_BYTES",
"SERVER_TCP_FLAGS",
"NUM_PKTS_512_TO_1024_BYTES",
"NUM_PKTS_256_TO_512_BYTES"
]
    
contaminants={}
contaminants["NF-CSE-CIC-IDS2018-V2"]=["MIN_TTL","SERVER_TCP_FLAGS"]

contaminants["NF-CSE-CIC-IDS2018-V2_generalization"]=generalizaion_contaminants

contaminants["NF-BoT-IoT-V2"]=['MIN_IP_PKT_LEN']
contaminants["NF-BoT-IoT-V2_generalization"]=generalizaion_contaminants

contaminants["NF-ToN-IoT-V2"]=["TCP_WIN_MAX_IN"]
contaminants["NF-ToN-IoT-V2_generalization"]=generalizaion_contaminants

contaminants["NF-UNSW-NB15-V2"]=["MIN_TTL", "MIN_IP_PKT_LEN", "TCP_WIN_MAX_OUT", "TCP_WIN_MAX_IN", 'SERVER_TCP_FLAGS']
contaminants["NF-UNSW-NB15-V2_generalization"]=generalizaion_contaminants

    
  
    
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
   
def drop_features(dataset,contaminant):
    if contaminant:
        for feature in contaminants[k]:
            cor=[feature_group for feature_group in correlations[dataset] if feature in feature_group]
            if len(cor)>0:
                for el in cor[0]:
                    try:
                        datasets[dataset]["X_train"].drop(columns=[el],inplace=True)
                    except KeyError:
                        print(f"{el} was already dropped")
            else:
                try:
                    datasets[dataset]["X_train"].drop(columns=[feature],inplace=True)
                except KeyError:
                    print(f"{feature} was already dropped")  
    else:
        for feature in contaminants[k+"_generalization"]:
            cor=[feature_group for feature_group in correlations[dataset] if feature in feature_group]
            if len(cor)>0:
                for el in cor[0]:
                    try:
                        datasets[dataset]["X_train"].drop(columns=[el],inplace=True)
                    except KeyError:
                        print(f"{el} was already dropped")
            else:
                try:
                    datasets[dataset]["X_train"].drop(columns=[feature],inplace=True)
                except KeyError:
                    print(f"{feature} was already dropped")
                    
def make_plots(attack, model, name):
    
    fig, ax = plt.subplots(figsize=(15, 15))

    stacked_array_norm = np.stack([row[2] for row in prepared_result_rows])
    stacked_array_norm=stacked_array_norm.reshape(4,-1)
    # create heatmap with annotations
    sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

    # set title and labels
    ax.set_title(f"inter-dataset generalization {attack} accuracy {name} dropped")
    ax.set_xlabel('test')
    ax.set_ylabel('train')

    # set x and y ticklabels to categorical labels
    ax.set_xticklabels(dataset_names)
    ax.set_yticklabels(dataset_names, rotation=0, ha='right')
    plt.subplots_adjust(left=0.2)
    plt.savefig(f"/project_ghent/raman/images/accuracy_heatmap_{attack}_{model}_{name}_dropped.png")


    fig, ax = plt.subplots(figsize=(15, 15))

    stacked_array_norm = np.stack([row[4] for row in prepared_result_rows])
    stacked_array_norm=stacked_array_norm.reshape(4,-1)
    # create heatmap with annotations
    sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

    # set title and labels
    ax.set_title(f"inter-dataset generalization {attack} precision {name} dropped")
    ax.set_xlabel('test')
    ax.set_ylabel('train')

    # set x and y ticklabels to categorical labels
    ax.set_xticklabels(dataset_names)
    ax.set_yticklabels(dataset_names, rotation=0, ha='right')
    plt.subplots_adjust(left=0.2)
    plt.savefig(f"/project_ghent/raman/images/precision_heatmap_{attack}_{model}_{name}_dropped.png")



    fig, ax = plt.subplots(figsize=(15, 15))

    stacked_array_norm = np.stack([row[6] for row in prepared_result_rows])
    stacked_array_norm=stacked_array_norm.reshape(4,-1)
    # create heatmap with annotations
    sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax)

    # set title and labels
    ax.set_title(f"Inter-dataset generalization {attack} recall {name} dropped")
    ax.set_xlabel('test')
    ax.set_ylabel('train')

    # set x and y ticklabels to categorical labels
    ax.set_xticklabels(dataset_names)
    ax.set_yticklabels(dataset_names, rotation=0, ha='right')
    plt.subplots_adjust(left=0.2)
    plt.savefig(f"/project_ghent/raman/images/recall_heatmap_{attack}_{model}_{name}_dropped.png")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 6))

    # Plot 1
    ax1 = axes[0]
    stacked_array_norm = np.stack([row[2] for row in prepared_result_rows])
    stacked_array_norm = stacked_array_norm.reshape(4, -1)
    sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax1)
    ax1.set_title(f"Inter-dataset generalization {attack} accuracy {name} dropped",y=1.1)
    ax1.set_xlabel('Test')
    ax1.set_ylabel('Train')
    ax1.set_xticklabels(dataset_names,rotation=-45)
    ax1.set_yticklabels(dataset_names, rotation=0, ha='right')
    plt.subplots_adjust(wspace=0.4)

    # Plot 2
    ax2 = axes[1]
    stacked_array_norm = np.stack([row[4] for row in prepared_result_rows])
    stacked_array_norm = stacked_array_norm.reshape(4, -1)
    sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax2)
    ax2.set_title(f"Inter-dataset generalization {attack} precision {name} dropped",y=1.1)
    ax2.set_xlabel('Test')
    ax2.set_ylabel('Train')
    ax2.set_xticklabels(dataset_names,rotation=-45)
    ax2.set_yticklabels(dataset_names, rotation=0, ha='right')

    # Plot 3
    ax3 = axes[2]
    stacked_array_norm = np.stack([row[6] for row in prepared_result_rows])
    stacked_array_norm = stacked_array_norm.reshape(4, -1)
    sns.heatmap(stacked_array_norm, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True, square=True, linewidths=.5, ax=ax3)
    ax3.set_title(f"Inter-dataset generalization {attack} recall {name} dropped",y=1.1)
    ax3.set_xlabel('Test')
    ax3.set_ylabel('Train')
    ax3.set_xticklabels(dataset_names,rotation=-45)
    ax3.set_yticklabels(dataset_names, rotation=0, ha='right')

    # Save the combined plot
    plt.savefig(f"/project_ghent/raman/images/combined_heatmaps_{attack}_{model}_{name}_dropped.png")


prepared_result_rows=inter_dataset()

make_plots("DoS", "knn", "_")

print("drop contaminants")
for k,v in datasets.items():
    print(datasets[k]["X_train"].shape)
    drop_features(k,True)
    print(datasets[k]["X_train"].shape)

prepared_result_rows=inter_dataset()

make_plots("DoS", "knn", "contaminants")

print("also drop generalization zone contaminants")
for k,v in datasets.items():
    print(datasets[k]["X_train"].shape)
    drop_features(k,False)
    print(datasets[k]["X_train"].shape)
    
prepared_result_rows=inter_dataset()

make_plots("DoS", "knn", "generalization contaminants")