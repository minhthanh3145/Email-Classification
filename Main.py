import numpy as np
import DataProcessingUtility as dataUtility
import os.path
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

def readData():    
    # Create a dictionary of words with its frequency
    train_dir = 'dataset/ling-spam/train-mails'
    dictionary = dataUtility.make_Dictionary(train_dir)
    
    # Prepare feature vectors per training mail and its labels
    train_labels = np.zeros(702)
    train_labels[351:701] = 1

    # Check if feature matrix file already exists
    train_matrix_file = Path('train_matrix.csv')
    if os.path.isfile(train_matrix_file):
        print("Reading training feature matrix from local cache")
        train_matrix = np.array(dataUtility.readMatrixFromFile('train_matrix.csv'))
    else:
        print("Training feature matrix doesn't exist, creating and writing into local cache")
        train_matrix = dataUtility.extract_features(train_dir, dictionary)            
        # Write feature matrix to file
        dataUtility.writeMatrixToFile(train_matrix, 'train_matrix.csv')

    
    # Prepare test data and labels
    test_dir = 'dataset/ling-spam/test-mails'
    
    # Check if feature matrix file already exists
    train_matrix_file = Path('test_matrix.csv')
    if os.path.isfile(train_matrix_file):
        print("Reading test feature matrix from local cache")
        test_matrix = np.array(dataUtility.readMatrixFromFile('test_matrix.csv'))
    else:
        print("Training test matrix doesn't exist, creating and writing into local cache")
        test_matrix = dataUtility.extract_features(test_dir, dictionary)
        # Write feature matrix to file
        dataUtility.writeMatrixToFile(test_matrix, 'test_matrix.csv')
    
    test_labels = np.zeros(260)
    test_labels[130:260] = 1
    return train_matrix, train_labels, test_matrix, test_labels


if __name__ == '__main__':
    
    train_matrix, train_labels, test_matrix, test_labels = readData()
    
    print("Training SVM and Naive Bayes model")
    # Train SVM and Naive Bayes
    print("Training SVM and Naive Bayes classifier")
    model1 = LinearSVC()
    model2 = MultinomialNB()    
    model1.fit(train_matrix,train_labels)
    model2.fit(train_matrix,train_labels)
    
    # Train Neural network
    print("Training Neural Network")
    model3 = MLPClassifier()
    model3.fit(train_matrix, train_labels)

    # Predict with SVM and Naive Bayes
    result1 = model1.predict(test_matrix)
    result2 = model2.predict(test_matrix)
    result3 = model3.predict(test_matrix)
    
    print("Perform PCA to reduce dimensionality")
    # Perform dimensionality reduction on train_matrix with PCA
    # PCA will choose N features such that more than 80% of the variance are explained
    pca1 = PCA(0.85)
    normalized_train_matrix = StandardScaler().fit_transform(train_matrix)
    pca1.fit(normalized_train_matrix)
    reduced_train_matrix = pca1.transform(normalized_train_matrix)
    
    print("Number of features before reduction", len(train_matrix[0]))
    print("Number of features after reduction", len(reduced_train_matrix[0]))

    pca2 = PCA(len(reduced_train_matrix[0]))
    normalized_test_matrix = StandardScaler().fit_transform(test_matrix)
    pca2.fit(normalized_test_matrix)
    reduced_test_matrix = pca2.transform(normalized_test_matrix)
    
    print("Training SVM and Neural Network on the reduced feature space")
    # After reduction
    model1.fit(reduced_train_matrix,train_labels)
    model3.fit(reduced_train_matrix,train_labels)
    
    result4 = model1.predict(reduced_test_matrix)
    result6 = model3.predict(reduced_test_matrix)
    
    # Result
    print("---------- Original feature space -------------")
    print("Confusion matrix for SVM")
    print(confusion_matrix(test_labels, result1))
    print("Confusion matrix for Naive Bayes")
    print(confusion_matrix(test_labels, result2))
    print("Confusion matrix for Neural network")
    print(confusion_matrix(test_labels, result3))
    print("---------- Reduced feature space -------------")
    print("Confusion matrix for SVM")
    print(confusion_matrix(test_labels, result4))
    print("Confusion matrix for Neural network")
    print(confusion_matrix(test_labels, result6))
    
    pass
