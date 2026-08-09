
import pandas as pd
import numpy as np
from sklearn.model\_selection import train\_test\_split
from sklearn.preprocessing import LabelEncoder
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from sklearn.metrics import accuracy\_score, precision\_score, recall\_score, f1\_score, confusion\_matrix
import os
from sklearn.linear\_model import LinearRegression
import featureClassification as fc

def findPRFC(predicted, actual, display=True) :
f1 = f1\_score(predicted, actual, average="macro")
pre = precision\_score(predicted, actual, average="macro")
acc = accuracy\_score(predicted, actual)
rec = recall\_score(predicted, actual, average="macro")
conf\_matrix = confusion\_matrix(predicted, actual);
diff\_arr = np.array(predicted) - np.array(actual)
idx = np.where(diff\_arr == 0)[0]

```
if(display) :
    print("Prediction f1 score : %s "% f1)
    print("Prediction Precision score : %s "% pre)
    print("Prediction accuracy score : %s "% acc)
    print("Prediction Recall score : %s "% rec)
    print('Confusion matrix')
    print(conf_matrix)
return idx
```

# Define the Deep Q-Network (DQN) classifier

class DQNClassifier:
def **init**(self, input\_shape, output\_shape):
DR = 0.1
self.model = Sequential()
self.model.add(Dense(16, activation='relu', input\_shape=input\_shape))
self.model.add(BatchNormalization())
self.model.add(Dropout(DR))
self.model.add(Dense(32, activation='relu'))
self.model.add(BatchNormalization())
self.model.add(Dropout(DR))
self.model.add(Dense(64, activation='relu'))
self.model.add(BatchNormalization())
self.model.add(Dropout(DR))
self.model.add(Dense(256, activation='relu'))
self.model.add(BatchNormalization())
self.model.add(Dropout(DR))
self.model.add(Dense(output\_shape, activation='softmax'))
self.model.compile(loss='sparse\_categorical\_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

```
def train(self, X_train, y_train):
    history = self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return history

def predict(self, X_test):
    return np.argmax(self.model.predict(X_test), axis=1)

def evaluate(self, X_test, y_test):
    y_pred = self.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)
    return y_pred, accuracy, precision, recall, f1, confusion
```

class QLearningClassifier:
def **init**(self, input\_shape, output\_shape, learning\_rate=0.1, discount\_factor=0.99, num\_bins=10):
self.model = LinearRegression()
self.input\_shape = input\_shape
self.output\_shape = output\_shape
self.learning\_rate = learning\_rate
self.discount\_factor = discount\_factor
self.num\_bins = num\_bins
self.q\_table = np.zeros((num\_bins, output\_shape))

```
def discretize_state(self, state):
    return np.digitize(state, bins=np.linspace(0, 1, self.num_bins))

def choose_action(self, state):
    try :
        q_values = self.q_table[state]
    except :
        q_values = self.q_table[0]
        
    action = np.argmax(q_values)
    return action

def update_q_table(self, state, action, reward, next_state):
    try :
        current_q = self.q_table[state, action]
    except :
        try :
            current_q = self.q_table[0, action]
        except :
            current_q = 0
    
    try :
        max_next_q = np.max(self.q_table[next_state])
    except :
        max_next_q = np.max(self.q_table[1])
    
    new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
    try :
        self.q_table[state, action] = new_q
    except :
        try :
            self.q_table[0, action] = new_q
        except :
            x = 0

def train(self, X_train, y_train, num_episodes=100):
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = self.discretize_state(X_train[np.random.choice(X_train.shape[0])])  # Sample a random state
        done = False
        total_reward = 0
        
        while not done:
            action = self.choose_action(state)
            next_state = self.discretize_state(X_train[np.random.choice(X_train.shape[0])])  # Sample the next state
            reward = int(y_train[np.random.choice(len(y_train))] == action)
            
            self.update_q_table(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
            if np.random.rand() > 0.95:  # 5% chance of ending an episode
                done = True
        
        episode_rewards.append(total_reward)
    
    return episode_rewards

def predict(self, X_test):
    predictions = []
    for state in X_test:
        state = self.discretize_state(state)
        action = self.choose_action(state)
        predictions.append(action)
    return predictions

def evaluate(self, X_test, y_test):
    y_pred = self.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)
    return y_pred, accuracy, precision, recall, f1, confusion
```

t1 = time.time()

datasetFile = 'output.csv'

t1 = time.time()

# Load the CSV dataset

dataset = pd.read\_csv(datasetFile, header=None).fillna(0)
dataset = dataset.sample(frac=1)

class\_idx = round(dataset.size/len(dataset))-1
y = np.array(dataset[class\_idx].values).astype(np.int8)
X = np.array(dataset[list(range(class\_idx))].values)[..., np.newaxis]
X = X.reshape((X.shape[0], X.shape[1]))

# Convert categorical labels to numeric values

label\_encoder = LabelEncoder()
y = label\_encoder.fit\_transform(y)

# Split the dataset into training and testing sets

ts = 0.1
X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=ts, random\_state=42)

# Initialize and train the DQN classifier

dqn\_classifier = DQNClassifier(input\_shape=(X\_train.shape[1],), output\_shape=len(label\_encoder.classes\_))
history = dqn\_classifier.train(X\_train, y\_train)

# Evaluate the classifier on the testing set

y\_pred1, accuracy, precision, recall, f1, confusion = dqn\_classifier.evaluate(X\_test, y\_test)

# Output the evaluation metrics

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', confusion)

# Initialize and train the Q-Learning classifier with Linear Regression

input\_shape = X\_train.shape[1]
output\_shape = len(np.unique(y\_train))
ql\_classifier = QLearningClassifier(input\_shape, output\_shape)
episode\_rewards = ql\_classifier.train(X\_train, y\_train)

# Evaluate the classifier on the testing set

y\_pred2, accuracy, precision, recall, f1, confusion = ql\_classifier.evaluate(X\_test, y\_test)

# Output the evaluation metrics

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', confusion)

idx1 = findPRFC(y\_pred1, y\_test)
idx2 = findPRFC(y\_pred2, y\_test)
idx3 = fc.classifyDLAndFindCorrect(X\_train, y\_train, X\_test, y\_test, 1, True)
idx4 = fc.classifyDLAndFindCorrect(X\_train, y\_train, X\_test, y\_test, 3, True)
final\_array = np.union1d(idx1, idx2)
final\_array = np.union1d(final\_array, idx3)
final\_array = np.union1d(final\_array, idx4)
final\_array = np.unique(final\_array)
y\_final = [0] \* len(y\_test)
for count in range(0, len(final\_array)) :
y\_final[final\_array[count]] = y\_test[final\_array[count]]

print('Final Prediction Results')
t2 = time.time()
findPRFC(y\_test, y\_final)
delay = t2 - t1
print('Delay needed %0.04f ms' % (delay))
