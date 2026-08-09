def classifyMFO(XTrain, YTrain, XTest, YTest) :
    import numpy
    import pygad.nn
    
    # Preparing the NumPy array of the inputs.
    data_inputs = XTrain
    
    # Preparing the NumPy array of the outputs.
    data_outputs = YTrain
    
    # The number of inputs (i.e. feature vector length) per sample
    num_inputs = data_inputs.shape[1]
    # Number of outputs per sample
    num_outputs = 2
    
    HL1_neurons = 2
    
    # Building the network architecture.
    input_layer = pygad.nn.InputLayer(num_inputs)
    hidden_layer1 = pygad.nn.DenseLayer(num_neurons=HL1_neurons, previous_layer=input_layer, activation_function="relu")
    output_layer = pygad.nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer1, activation_function="softmax")
    
    # Training the network.
    pygad.nn.train(num_epochs=10,
                   last_layer=output_layer,
                   data_inputs=data_inputs,
                   data_outputs=data_outputs,
                   learning_rate=0.01)
    
    # Using the trained network for predictions.
    predictions = pygad.nn.predict(last_layer=output_layer, data_inputs=XTest)
    
    return predictions

def findVariantFeatures(features, classes, method) :
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC
    
    out_features = features
    
    if(method == 1) :
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(out_features, classes)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(out_features)
    else :
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(out_features, classes)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(out_features)
        
    return X_new

def applyDMFO(features, classes, method) :
    import pandas as pd
    import numpy as np
    import random
    import math
    from statsmodels.tsa.arima.model import ARIMA
    from tensorflow import keras
    
    from tensorflow.keras.layers import Input, Embedding, GRU, LSTM, MaxPooling1D, GlobalMaxPool1D
    from tensorflow.keras.layers import Dropout, Dense, Activation, Flatten,Conv1D, SpatialDropout1D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import RMSprop, SGD
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    
    def predictDataARIMA(array, numElements) :
        X = array
        
        while(len(X) < numElements) :
            model = ARIMA(X)
            model_fit = model.fit()
            output = model_fit.forecast()
            X.append(output[0])
        
        return X
    
    def findAverage(array) :
        avg = 0
        for count in range(0, len(array)) :
            ele = array[count]
            avg = avg + findArraySum(ele)
            
        return avg / len(array)
    
    def applyGRUCNN(train_data, classes, epochs, batch_size) :
        X_train = np.asarray(train_data)
        X_train = np.reshape(X_train, X_train.shape + (1,))
        y_train = np.asarray(classes)
        # define model architecture
        regressorGRU = Sequential()
        # First GRU layer with Dropout regularisation
        regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
        regressorGRU.add(Dropout(0.2))
        # Second GRU layer
        regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
        regressorGRU.add(Dropout(0.2))
        # Third GRU layer
        regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
        regressorGRU.add(Dropout(0.2))
        # Fourth GRU layer
        regressorGRU.add(GRU(units=50, activation='tanh'))
        regressorGRU.add(Dropout(0.2))
        # The output layer
        regressorGRU.add(Dense(units=1))
        # Compiling the RNN
        regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
        early_stop = keras.callbacks.EarlyStopping(monitor = 'loss',patience = 10)
        
        # Fitting to the training set
        history = regressorGRU.fit(X_train,y_train,epochs=epochs,batch_size=batch_size, callbacks = [early_stop])
        return (history, regressorGRU)
    
    def findNumElementsInArray(array, element) :
        x = sum(1 for i in array if i == element);
        return x
    
    def findArraySum(array) :
        return sum(ord(i) for i in array);
    
    def findVariance(array, element) :
        sum_arr = findArraySum(array)
        avg_arr = sum_arr / len(array)
        sum_val = 0
        
        for count in range(0,len(element)) :
            ele = ord(element[count])
            sum_val = sum_val + pow(ele-avg_arr,2)
        
        sum_val = sum_val / (len(array)-1)
        var = math.sqrt(sum_val)
        
        return var
    
    try :
            
        #Step 1. Read dataset
        dataset = pd.read_csv(datasetFile, header=None)
        dataset = dataset.sample(frac=1)
        
        df_train = pd.read_csv(datasetFile, header=None)
        df_train = df_train.sample(frac=1)
        
        class_idx = int(df_train.size/len(df_train))
        class_idx = class_idx - 1
        
        Y = np.array(df_train[class_idx].values).astype(np.int8)
        X = np.array(df_train[list(range(class_idx))].values)[..., np.newaxis]
        X = X.reshape((X.shape[0], X.shape[1]))
        
        #Step 2. Segregate into features and classes
        features = dataset[0]
        classes = dataset[class_idx]
        unique_classes = np.unique(classes)
        num_classes = len(unique_classes)
        
        #Step 3. Initialize input parameters
        Nmoths = 3
        Ni = 3
        Mu = 0.95
        
        #Find max samples per class in the training set
        max_samples_per_class = [0] * num_classes
        for count in range(0, num_classes) :
            max_samples_per_class[count] = findNumElementsInArray(classes, unique_classes[count])
            max_samples_per_class[count] = round(max_samples_per_class[count] * Tr)
        
        #Part 1. Train and test set selection
        sols_to_modify = [1] * Nmoths
        fitness = [0] * Nmoths
        sol_train_sets = [0] * Nmoths
        sol_train_classes = [0] * Nmoths
        
        CVV = 0
        
        for iter_val in range(0,Ni):
            
            for sol_val in range(0,Nmoths):
                
                if(sols_to_modify[sol_val] == 1) :
                    
                    #Put random values into the train set
                    train_set = []
                    train_classes = []
                    
                    for count in range(0,num_classes) :
                        #Get the class value
                        class_val = unique_classes[count]
                        #Find number of elements in this array (initially 0)
                        num_elements = findNumElementsInArray(train_classes, class_val)
                        
                        #Loop till the array is filled
                        counter2 = 0
                        while(num_elements <= max_samples_per_class[count] and counter2 <= 10) :
                            #Get a random feature vector
                            idx = random.randint(0, len(features)-1)
                            counter2 = counter2 + 1
                            
                            #Check if this feature is already present in the set
                            if(classes[idx] == class_val and not(features[idx] in train_set)) :
                                #If not then add it
                                train_set.append(features[idx])
                                train_classes.append(class_val)
                                print('Class %d, %0.04f' % (count, num_elements*100/max_samples_per_class[count]))
                            
                            num_elements = findNumElementsInArray(train_classes, class_val)
                    
                    #Now find variance between different classes
                    var_vals = []
                    
                    # for count in range(0, num_classes) :
                    #     var_val = 0
                    #     var_count = 0
                        
                    #     class_val = unique_classes[count]
                    #     for count2 in range(0, len(train_set)):
                    #         class_val2 = train_classes[count2]
                            
                    #         if(class_val == class_val2) :
                    #             sequence = train_set[count2]
                    #             other_arr = []
                    #             for count3 in range(0, len(train_set)) :
                    #                 if(class_val != train_classes[count3]) :
                    #                     other_arr.append(train_set[count3])
                                        
                    #             for count4 in range(0, len(other_arr)) :
                    #                 var_val = var_val + findVariance(other_arr[count4], sequence)
                    #                 var_count = var_count + 1
                        
                    #     var_val = var_val / var_count
                    #     var_vals.append(var_val)
                    try :
                        
                        avg_val = findAverage(train_set)
                        
                        for count in range(0, len(train_set)) :
                            sequence = train_set[count]
                            var_val = 0
                            var_count = 0
                            print('Processing:%0.02f' % (count*100/len(train_set)))
                        
                            for count2 in range(0, len(sequence)) :
                                var_val = var_val + pow(ord(sequence[count2]) - avg_val, 2)
                                var_count = var_count + 1
                                
                            var_val = math.sqrt(var_val) / var_count
                            var_vals.append(var_val)
                        
                        fitness[sol_val] = sum(var_vals) / len(var_vals)
                        sol_train_sets[sol_val] = train_set
                        sol_train_classes[sol_val] = train_classes
                        print('Moth %d processed...' % (sol_val))
                    except :
                        continue;
                    
            avg_fitness = sum(fitness)/len(fitness)
            fitness_th = avg_fitness * Mu
            for count in range(0,len(fitness)) :
                if(fitness[count] < fitness_th) :
                    sols_to_modify[count] = 1
                else :
                    sols_to_modify[count] = 0
                    
        max_fitness = max(fitness)
        idx = fitness.index(max_fitness)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1-Tr, random_state=0)
        sel_train_set = sol_train_sets[idx]
        sel_train_classes = sol_train_classes[idx]
        max_len = 0
        min_len = 0
        try :
            for count in range(0,len(sel_train_set)) :
                len_val = len(sel_train_set[count])
                if(count == 0) :
                    max_len = len_val
                    min_len = len_val
                else :
                    if(len_val > max_len) :
                        max_len = len_val;
                        
                    if(len_val < min_len) :
                        min_len = len_val
                
            print('Max Length %d, Min Length %d' % (max_len, min_len))
            
            #Part 2 Feature selection for effective classification
            sols_to_modify = [1] * Nmoths
            fitness = [0] * Nmoths
            sol_train_sets = [0] * Nmoths
            sol_train_classes = [0] * Nmoths
            CA = 0
            
            #Convert data to integers
            sel_train_set_int = []
            for count in range(0, len(sel_train_set)) :
                train_data = sel_train_set[count]
                train_set = []
                for count2 in range(0, len(train_data)) :
                    train_set.append(ord(train_data[count2]))
                sel_train_set_int.append(train_set)
            
            models = [0] * Nmoths
            
            for iter_val in range(0, Ni) :
                for sol_val in range(0, Nmoths) :
                    
                    if(sols_to_modify[sol_val] == 1) :
                        sl_sel = random.randint(min_len, max_len)
                        
                        #Create a solultion training set
                        sol_train_set = []
                        for count in range(0, len(sel_train_set_int)) :
                            print('Selecting %0.04f %%' % (count*100/len(sel_train_set_int)))
                            
                            if(len(sel_train_set_int[count]) <= sl_sel) :
                                selected_seq = predictDataARIMA(sel_train_set_int[count], sl_sel)
                            else :
                                selected_seq = sel_train_set_int[count][0:sl_sel]
                            
                            sol_train_set.append(selected_seq)
                        
                        cnn_train_set = sol_train_set
                        cnn_train_classes = sel_train_classes
                        
                        history, model = applyGRUCNN(cnn_train_set, cnn_train_classes, 10, 32)
                        acc_val = 1-min(history.history['loss']);
                        models[sol_val] = model
                        fitness[sol_val] = acc_val
                        
                        if(acc_val >= CA) :
                            CA = acc_val
                            sel_sol = sol_val
                    
                sols_to_modify = [1] * Nmoths
                sols_to_modify[sel_sol] = 0
                
            max_acc = max(fitness)
            max_idx = fitness.index(max_acc)
            model = models[max_idx]
        except :
            model = 1
            
        return X_train, X_test, y_train, y_test
    except :
        return findVariantFeatures(features, classes, method)
