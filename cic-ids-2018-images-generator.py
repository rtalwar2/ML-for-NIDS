
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
print("importing stuff")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
dfps = []
# for dirname, _, filenames in os.walk('/project_ghent/datasets/clean-ids-collection/cic-ids2017/clean'):
for dirname, _, filenames in os.walk('/project_ghent/datasets/clean-ids-collection/cse-cic-ids2018/clean'):
    for filename in filenames:
        if filename.endswith('.parquet'):
            dfp = os.path.join(dirname, filename)
            dfps.append(dfp)
            print(dfp)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from fastai.imports import *
from fastai.vision.all import *
from sklearn.preprocessing import MinMaxScaler
dfps = [dfp for dfp in dfps if not 'Benign' in dfp] #remove Benign monday
df = pd.concat([pd.read_parquet(dfp) for dfp in dfps], ignore_index=True)

df.loc[df['Label'] != 'Benign', 'Label'] = 1
df.loc[df['Label'] == 'Benign', 'Label'] = 0
df['Label'] = df['Label'].astype(dtype=np.int32)

def xs_y(df_, targ): 
    if not isinstance(targ, list):
        xs = df_[df_.columns.difference([targ])].copy() #splot target from data
    else:
        xs = df_[df_.columns.difference(targ)].copy()
    y = df_[targ].copy()
    return xs, y
print("splitting train and test set")
training_set = df.sample(frac=0.2, replace=False, random_state=42)
testing_set = df.drop(index=training_set.index)
training_set.shape, testing_set.shape

X_train, y_train = xs_y(training_set, targ="Label")
X_test, y_test = xs_y(testing_set, targ="Label")

df_norm=X_train.copy()

scaler=MinMaxScaler((0,255))
scaled_features=scaler.fit_transform(df_norm)
print("scaling")
df=pd.DataFrame(scaled_features,columns=df_norm.columns)

path = Path('/project_ghent/raman/cic-ids2018/flow_images/train')
labels=["benign","malign"]

def flow_to_image(i):
    np_image=np.concatenate((df.iloc[i],[0]*15)).reshape(9,9)
    im=PILImage.create(np_image.astype(np.uint8))
#     im=im.resize((256,256))
    #save the image in correct folder
    dest=(path/labels[y_train.iloc[i]]) #dest will be flow_images/benign or flow_images/malign
    dest.mkdir(exist_ok=True, parents=True)
    im.save(dest/f"{i}.jpg")
    print(i)
print("started with 2018 train")
parallel(f=flow_to_image, 
                  items=range(df.shape[0]), n_workers=os.cpu_count(), threadpool=False, progress=True)

df_norm=X_test.copy()
scaled_features=scaler.fit_transform(df_norm)
df=pd.DataFrame(scaled_features,columns=df_norm.columns)

path = Path('/project_ghent/raman/cic-ids2018/flow_images/test')

def flow_to_image(i):
    np_image=np.concatenate((df.iloc[i],[0]*15)).reshape(9,9)
    im=PILImage.create(np_image.astype(np.uint8))
    #save the image in correct folder
    dest=(path/labels[y_test.iloc[i]]) #dest will be flow_images/benign or flow_images/malign
    dest.mkdir(exist_ok=True, parents=True)
    im.save(dest/f"{i}.jpg")
    print(i)
print("started with 2018 test")
parallel(f=flow_to_image, 
                  items=range(df.shape[0]), n_workers=os.cpu_count(), threadpool=False, progress=True)