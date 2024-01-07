import numpy as np
import scipy.io
import pandas as pd

dataset_name = 'air_datasets'
dataset_specific = 'china_air'
file_name = 'data/data_detective/{}/{}.mat'.format(dataset_name,dataset_specific)
print(file_name)
data = scipy.io.loadmat(file_name)
adj_name = 'State_adj'
y_key = 'Y5'
x_data = np.array(data['X'],dtype=np.float16)  # shape (26,550,1289) (Task ID (Spatial location), Sample (Temporal-Days), Features(TF-IDF)) # Civil Unrest Data
y_data = np.array(data[y_key],dtype=np.float16)  # shape (26, 550) # Y5

# Making lead data
lead_time = 1
y_data = y_data[:,lead_time:]
x_data = x_data[:,:-lead_time,:]

print("Number of subtypes: {}".format(np.unique(y_data-1)))
adj_data = np.array(data[adj_name],dtype=np.float16)  # shape (26, 26)
DF = pd.DataFrame(adj_data)
DF.to_csv("data/{}_adj.csv".format(dataset_specific), index=False, header=False)
y_data_3d = np.expand_dims(y_data, axis=2)
data_concat = np.concatenate((x_data, y_data_3d), axis=2)
DF_feat = pd.DataFrame(np.swapaxes(data_concat.reshape(data_concat.shape[0],-1), 0, 1))
# DF_feat = pd.DataFrame(np.swapaxes(y_data, 0, 1))
DF_feat.to_csv("data/{}_feat.csv".format(dataset_specific), index=False, header=True)
DF_Y = pd.DataFrame(np.swapaxes(y_data-1, 0, 1))
DF_Y.to_csv("data/{}_y.csv".format(dataset_specific), index=False, header=True)
print("Data Saved")