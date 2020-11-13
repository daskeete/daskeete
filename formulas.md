def summarizedata(x):
    #'''function takes list or array as input, and output a dataframe with some summary statistics'''
    import numpy as np
    import pandas as pd
    #compute summary stats
    #data points
    len_x = len(x)
    #mean
    mean_x = np.mean(x)

    #median
    median_x = np.median(x)
    #std
    std_x = np.std(x)
    #maximum
    max_x = np.max(x)
    #minimum
    min_x= np.min(x)
    #coefficient of variation
    cv_x = std_x/mean_x

    #Create df

    summary = pd.DataFrame(columns=['N', 'Min', 'Max', 'Mean', 'Median', 'Std', 'Cv'])

    data = [len_x, min_x, max_x, mean_x, median_x, std_x, cv_x]

    #insert list as first row as df

    summary.loc[0] = data

    return summary

def conf_matrix(actual, predicted):
    import numpy as np
    import pandas as pd
    classes = pd.DataFrame(columns=['Actual','Predicted'])
    #y_names = np.unique(y)
    #actual.index=y_names
    classes['Actual'] = actual
    classes['Predicted'] = predicted
    a = pd.crosstab(classes['Predicted'],classes['Actual'])
    return a

def summarizedata(x):
    #'''function takes list or array as input, and output a dataframe with some summary statistics'''
    import numpy as np
    import pandas as pd
    #compute summary stats
    #data points
    len_x = len(x)
    #mean
    mean_x = np.mean(x)

    #median
    median_x = np.median(x)
    #std
    std_x = np.std(x)
    #maximum
    max_x = np.max(x)
    #minimum
    min_x= np.min(x)
    #coefficient of variation
    cv_x = std_x/mean_x

    #Create df

    summary = pd.DataFrame(columns=['N', 'Min', 'Max', 'Mean', 'Median', 'Std', 'Cv'])

    data = [len_x, min_x, max_x, mean_x, median_x, std_x, cv_x]

    #insert list as first row as df

    summary.loc[0] = data

    return summary

def conf_matrix(actual, predicted):
    import numpy as np
    import pandas as pd
    classes = pd.DataFrame(columns=['Actual','Predicted'])
    #y_names = np.unique(y)
    #actual.index=y_names
    classes['Actual'] = actual
    classes['Predicted'] = predicted
    a = pd.crosstab(classes['Predicted'],classes['Actual'])
    return a

def do_logistic_regression(xtrain, ytrain, xtest, ytest, scale=None):
    ### This function returns ypred and prints the accuracy_score, pass in scaler as a string using full name with no spaces###

    from sklearn.linear_model import LogisticRegression
    logistic_regression = LogisticRegression(max_iter=500000)

    if scale == 'StandardScaler':
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()
        xtrain = ss.fit_transform(xtrain)
        xtest = ss.transform(xtest)
    if scale == 'MinMaxScaler':
        from sklearn.preprocessing import MinMaxScaler as MMS
        mms = MMS()
        xtrain = mms.fit_transform(xtrain)
        xtest = mms.transform(xtest)
    if scale == 'RobustScaler':
        from sklearn.preprocessing import RobustScaler as RS
        rs = RS()
        xtrain = rs.fit_transform(xtrain)
        xtest = rs.transform(xtest)
    if scale == 'Normalizer':
        from sklearn.preprocessing import Normalizer as NMZ
        nmz = NMZ()
        xtrain = nmz.fit_transform(xtrain)
        xtest = nmz.transform(xtest)

    logistic_regression.fit(xtrain, ytrain)
    ypred = logistic_regression.predict(xtest)

    from sklearn.metrics import accuracy_score
    #a = accuracy_score(ytest,ypred)
    #print(a)
    a = accuracy_score(ytest,ypred)
    b = logistic_regression.score(xtrain,ytrain)
    print('accuracy score test ->',a,'acc_train','->',b)
    #return ypred, acc_test, acc_train
    return ypred


def do_knn(xtrain, ytrain, xtest, ytest,n, scale=None):
    ### This function returns ypred and prints the accuracy_score, pass in scaler as a string using full name with no spaces###

    from sklearn.neighbors import KNeighborsClassifier as KNN
    knn = KNN(n_neighbors=n)

    if scale == 'StandardScaler':
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()
        xtrain = ss.fit_transform(xtrain)
        xtest = ss.transform(xtest)
    if scale == 'MinMaxScaler':
        from sklearn.preprocessing import MinMaxScaler as MMS
        mms = MMS()
        xtrain = mms.fit_transform(xtrain)
        xtest = mms.transform(xtest)
    if scale == 'RobustScaler':
        from sklearn.preprocessing import RobustScaler as RS
        rs = RS()
        xtrain = rs.fit_transform(xtrain)
        xtest = rs.transform(xtest)
    if scale == 'Normalizer':
        from sklearn.preprocessing import Normalizer as NMZ
        nmz = NMZ()
        xtrain = nmz.fit_transform(xtrain)
        xtest = nmz.transform(xtest)

    model = knn.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    from sklearn.metrics import accuracy_score
    a = accuracy_score(ytest,ypred)

    acc_test = accuracy_score(ytest,ypred)
    acc_train = model.score(xtrain,ytrain)
    print('accuracy score ->',a, '----','test accuracy','->',acc_test,'----','acc_train','->',acc_train)
    return ypred, acc_test, acc_train

def find_best_n(start,stop,step,xtrain, ytrain, xtest, ytest,scale=None):
    import numpy as np
    i_range = np.arange(start,stop,step)
    best_accuracy_test = -np.inf
    kacctest = []
    kacctrain = []
    for i in i_range:
    #from sklearn.neighbors import KNeighborsClassifier as KNN
        #knn = KNN(n_neighbors=i)
        z = do_knn(xtrain, ytrain, xtest, ytest,n=i, scale=None)
        kacctest.append(z[1])
        kacctrain.append(z[2])
        #if z[1] > np.any(best_accuracy_test):
        if z[1] > best_accuracy_test:
            best_accuracy_test = z[1]
            k = i
    print('best_accuracy_test -->',best_accuracy_test,'----','k','-->',k)
    return best_accuracy_test, k,kacctest,kacctrain

def make_knn_plot(the_range, acc_test, acc_train):
    import matplotlib.pyplot as plt
    kacctest = acc_test
    kacctrain = acc_train
    i_range = the_range
    plt.plot(i_range,kacctest,'-xk', label='Testing')
    plt.plot(i_range,kacctrain,'-xr', label='Train')
    plt.xlabel('$k$')
    plt.ylabel('Fraction correctly classified')
    plt.legend()
    plt.show()

def do_dtr(rand_s, d, xtrain, ytrain, xtest, ytest,scale=None):
    from sklearn.tree import DecisionTreeRegressor as DTR
    from sklearn import tree

    dt = DTR(random_state=rand_s,max_depth=d)
    #dt.fit(Xtrain,ytrain)
    #ypred = dt.predict(Xtest)

    return dt






def DoKFold(model,X,y,k,random_state=146,scaler=None):
    '''Function will perform K-fold validation and return a list of K training and testing scores, inclduing R^2 as well as MSE.

        Inputs:
            model: An sklearn model with defined 'fit' and 'score' methods
            X: An N by p array containing the features of the model.  The N Columns are features, and the p rows are observations.
            y: An array of length N containing the target of the model
            k: The number of folds to split the data into for K-fold validation
            random_state: used when splitting the data into the K folds (default=146)
            scaler: An sklearn feature scaler.  If none is passed, no feature scaling will be performed
        Outputs:
            train_scores: A list of length K containing the training scores
            test_scores: A list of length K containing the testing scores
            train_mse: A list of length K containing the MSE on training data
            test_mse: A list of length K containing the MSE on testing data
    '''

    from sklearn.model_selection import KFold
    import numpy as np
    import pandas as pd
    kf = KFold(n_splits=k,shuffle=True,random_state=random_state)

    train_scores=[]
    test_scores=[]
    train_mse=[]
    test_mse=[]

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain,:]
        Xtest = X[idxTest,:]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if scaler != None:
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)

        model.fit(Xtrain,ytrain)

        train_scores.append(model.score(Xtrain,ytrain))
        test_scores.append(model.score(Xtest,ytest))

        # Compute the mean squared errors
        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)
        train_mse.append(np.mean((ytrain-ytrain_pred)**2))
        test_mse.append(np.mean((ytest-ytest_pred)**2))

    return train_scores,test_scores,train_mse,test_mse
