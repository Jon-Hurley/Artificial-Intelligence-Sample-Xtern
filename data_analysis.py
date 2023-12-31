import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
import math
# Read the data from the csv file

def process_data():
    df = pd.read_csv('data.csv')


    # The orders are the labels
    # The rest of the data is the features

    y_labels = df['Order']

    X_data = df.drop('Order', axis=1)



    # Create numerical labels for the data

    y_train = y_labels.astype('category').cat.codes

    X_train = X_data.astype('category').apply(lambda x: x.cat.codes)


    # Split the data into training and testing data
    # 1000 data points for testing
    # 4000 data points for training
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    return X_train, X_test, y_train, y_test

def compute_statistics(X_train):

    # get statistical measures of the data for each feature
    print(f"X_train feature means: {X_train.mean()}")
    print(f"X_train feature std: {X_train.std()}")
    print(f"X_train feature max: {X_train.max()}")
    print(f"X_train feature min: {X_train.min()}")
    print(f"X_train feature median: {X_train.median()}")
    print(f"X_train feature mode: {X_train.mode()}")
    print(f"X_train feature variance: {X_train.var()}")
    print(f"X_train feature skew: {X_train.skew()}")
    print(f"X_train feature kurtosis: {X_train.kurtosis()}")
    print(f"X_train feature quantile: {X_train.quantile()}")
    print(f"X_train feature corr: {X_train.corr()}")
    print(f"X_train feature count: {X_train.count()}")

def plot_data(X_train, y_train):
    
        # plot the data
        plt.scatter(X_train['Year'], y_train)
        plt.xlabel('Year')
        plt.ylabel('Order')
        plt.show()
    
        plt.scatter(X_train['Major'], y_train)
        plt.xlabel('Major')
        plt.ylabel('Order')
        plt.show()
    
        plt.scatter(X_train['University'], y_train)
        plt.xlabel('University')
        plt.ylabel('Order')
        plt.show()
    
        plt.scatter(X_train['Time'], y_train)
        plt.xlabel('Time')
        plt.ylabel('Order')
        plt.show()

        # create normal distribution
        plt.hist(X_train['Year'], bins=math.ceil(math.sqrt(len(X_train['Year']))))
        plt.show()

        plt.hist(X_train['Major'], bins=math.ceil(math.sqrt(len(X_train['Major']))))
        plt.show()

        plt.hist(X_train['University'], bins=math.ceil(math.sqrt(len(X_train['University']))))
        plt.show()

        plt.hist(X_train['Time'], bins=math.ceil(math.sqrt(len(X_train['Time']))))
        plt.show()




# use a support vector machine model
def svm_model():

    X_train, X_test, y_train, y_test = process_data()

    clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=1.0)

    clf.fit(X_train, y_train)

    prediction = clf.predict([[1, 1, 1, 5]])
    # print(f"prediction: {prediction[0]}")

    # run the test data through the model and get the predictions for the test data
    y_pred = clf.predict(X_test)
    y_true = y_test

    # compute the accuracy of the model
    accuracy = accuracy_score(y_true, y_pred)
    print(f"svm accuracy: {accuracy}")
    return accuracy, y_pred


def knn_model():

    X_train, X_test, y_train, y_test = process_data()


    # use a k nearest neighbors model
    knn = KNeighborsClassifier(n_neighbors=10)
    plt.plot()
    knn.fit(X_train, y_train)
    prediction = knn.predict([[1, 1, 1, 5]])
    # print(f"knn prediction: {prediction[0]}")

    # run the test data through the model and get the predictions for the test data
    y_pred = knn.predict(X_test)
    y_true = y_test

    # compute the accuracy of the model
    accuracy = accuracy_score(y_true, y_pred)
    print(f"knn accuracy: {accuracy}")
    return accuracy, y_pred

def random_forest_model():
    X_train, X_test, y_train, y_test = process_data()

    # use a random forest model
    clf = RandomForestClassifier(max_depth=None, n_estimators=50, max_samples=0.5, max_features=0.5, random_state=0)
     
    clf.fit(X_train, y_train)
    prediction = clf.predict([[1, 1, 1, 5]])
    # print(f"random forest prediction: {prediction[0]}")

    # run the test data through the model and get the predictions for the test data
    y_pred = clf.predict(X_test)
    y_true = y_test

    # compute the accuracy of the model
    accuracy = accuracy_score(y_true, y_pred)
    print(f"random forest accuracy: {accuracy}")
    return accuracy, y_pred


def decision_tree_model():

    X_train, X_test, y_train, y_test = process_data()

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    prediction = clf.predict([[1, 1, 1, 5]])
    # print(f"tree prediction: {prediction[0]}")

    # run the test data through the model and get the predictions for the test data
    y_pred = clf.predict(X_test)
    y_true = y_test

    # compute the accuracy of the model
    accuracy = accuracy_score(y_true, y_pred)
    print(f"tree accuracy: {accuracy}")
    return accuracy, y_pred


# use a linear regression model
def linear_regression_model():

    X_train, X_test, y_train, y_test = process_data()

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    # print(reg.coef_)
    # print(np.mean((reg.predict(X_test) - y_test) ** 2))
    print(f"linear regression score: {reg.score(X_train, y_train)}")
    return reg.score(X_train, y_train), reg.predict(X_test)




# use a logistic regression model
def logistic_regression_model():

    X_train, X_test, y_train, y_test = process_data()

    log = linear_model.LogisticRegression()
    log.fit(X_train, y_train)
    print(f"logistic regression score: {log.score(X_train, y_train)}")
    # print(f"logistic regression prediction: {log.predict([[1, 1, 1, 5]])[0]}")
    return log.score(X_train, y_train), log.predict(X_test)

def pickler(clf, X_test):
    s = pickle.dumps(clf)
    clf2 = pickle.loads(s)
    print(clf2.predict(X_test))



# print(y_labels)
# print(X_data)
# Compute the mean of the data

def final_model():
    # pickle the model with the highest accuracy
    X_train, X_test, y_train, y_test = process_data()

    svm_accuracy, svm_pred = svm_model()
    knn_accuracy, knn_pred = knn_model()
    tree_accuracy, decision_tree_pred = decision_tree_model()
    random_forest_accuracy, random_forest_pred = random_forest_model()
    
    # pickle the model with the highest accuracy

    model_accuracies = [ knn_accuracy, tree_accuracy, random_forest_accuracy]
    max_accuracy = model_accuracies.index(max(model_accuracies))
    print(max_accuracy)
    if max_accuracy == 0:
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=1.0)
        clf.fit(X_train, y_train)
        pickler(clf, X_test)
        print("svm")
        return svm_pred
    elif max_accuracy == 1:
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        pickler(knn, X_test)
        print("knn")
        return knn_pred
    elif max_accuracy == 2:
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        pickler(clf, X_test)
        print("tree")
        #plot the decision tree
        plt.figure(figsize=(20, 20))
        tree.plot_tree(clf, filled=True, fontsize=5)
        plt.show()
        return decision_tree_pred
    elif max_accuracy == 3:
        clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=100)
        X_train, y_train = make_classification(n_samples=1000, n_features=4)
        clf.fit(X_train, y_train)
        pickler(clf, X_test)
        print("random forest")
        return random_forest_pred
def main():
    final_model()



if __name__ == '__main__':
    main()


