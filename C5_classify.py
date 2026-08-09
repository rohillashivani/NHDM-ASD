import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
import featureClassification as fc
from sklearn.model_selection import train_test_split

datasetFile = "output.csv"

df_train = pd.read_csv(datasetFile, header=None).fillna(0)
df_train = df_train.sample(frac=1)
class_idx = int(round(df_train.size/len(df_train)))-1

Y = np.array(df_train[class_idx].values).astype(np.int8)
X = np.array(df_train[list(range(class_idx))].values)[..., np.newaxis]
X = X.reshape((X.shape[0], X.shape[1]))

test_size = 0.5
mode = 'test'

t1 = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
if(mode == 'train') :
    X_test = X_train
    y_test = y_train
    
print('kNN...')
t1 = time.time()
arr1 = fc.classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 1)
t2 = time.time()
print('Delay:%0.04f s' % (t2-t1))
print('RF...')
t1 = time.time()
try :
    arr2 = fc.classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 2)
except :
    arr2 = []
    
t2 = time.time()
print('Delay:%0.04f s' % (t2-t1))
print('SVM...')
t1 = time.time()
try :
    arr3 = fc.classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 3)
except :
    arr3 = []
    
t2 = time.time()
print('Delay:%0.04f s' % (t2-t1))
print('LR...')
t1 = time.time()
try :
    arr4 = fc.classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 4)
except :
    arr4 = []
    
t2 = time.time()
print('Delay:%0.04f s' % (t2-t1))

t2 = time.time()

delay = t2 - t1

t1 = time.time()
final_array = np.union1d(arr1, arr2)
final_array = np.union1d(final_array, arr3)
final_array = np.union1d(final_array, arr4)

final_array = np.unique(final_array)
y_final = [0] * len(y_test)
for count in range(0, len(final_array)) :
    y_final[round(final_array[count])] = y_test[round(final_array[count])]
t2 = time.time()

delay = t2 - t1

fc.findPRFC(y_test, y_final, True)

print('Delay for classification:%0.04f s' %(delay))
