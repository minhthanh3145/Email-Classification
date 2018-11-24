import numpy as np
import DataProcessingUtility as dataUtility
import os.path
from NeuralNetworkClassifier import NeuralNetworkClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.decomposition import PCA

def readDataAndConstructFeatureMatrix():    
    # Create a dictionary of words with its frequency
    print("Create a dictionary of words with its frequency")
    train_dir = 'dataset/ling-spam/train-mails'
    dictionary = dataUtility.make_Dictionary(train_dir)
    
    # Prepare feature vectors per training mail and its labels
    print("Prepare feature vectors per training mail and its labels")
    train_labels = np.zeros(702)
    train_labels[351:701] = 1

    # Check if feature matrix file already exists
    train_matrix_file = Path('train_matrix.csv')
    if os.path.isfile(train_matrix_file):
        train_matrix = dataUtility.readMatrixFromFile('train_matrix.csv')
    else:  
        train_matrix = dataUtility.extract_features(train_dir, dictionary)            
        # Write feature matrix to file
        dataUtility.writeMatrixToFile(train_matrix, 'train_matrix.csv')

    
    # Prepare test data and labels
    print("Prepare feature vectors per testing mail and its labels")
    test_dir = 'dataset/ling-spam/test-mails'
    
    # Check if feature matrix file already exists
    train_matrix_file = Path('test_matrix.csv')
    if os.path.isfile(train_matrix_file):
        test_matrix = dataUtility.readMatrixFromFile('test_matrix.csv')
    else:
        test_matrix = dataUtility.extract_features(test_dir, dictionary)    
        # Write feature matrix to file
        dataUtility.writeMatrixToFile(test_matrix, 'test_matrix.csv')
    
    test_labels = np.zeros(260)
    test_labels[130:260] = 1
    return train_matrix, train_labels, test_matrix, test_labels



def trainSVMAndNaiveBayes(train_matrix, train_labels):
    # SVM and Naive Bayes 
    print("Training SVM and Naive Bayes classifier and its variants")
    model1 = LinearSVC()
    model2 = MultinomialNB()
    
    model1.fit(train_matrix,train_labels)
    model2.fit(train_matrix,train_labels)
    return model1, model2



if __name__ == '__main__':
    
    train_matrix, train_labels, test_matrix, test_labels = readDataAndConstructFeatureMatrix()
    
    print("Training SVM and Naive Bayes model")
    # Train SVM and Naive Bayes
    model1, model2 = trainSVMAndNaiveBayes(train_matrix, train_labels)
    
    print("Perform PCA on training feature matrix")
    # Perform dimensionality reduction on train_matrix with PCA
    pca1 = PCA(0.90)
    normalized_train_matrix = StandardScaler().fit_transform(train_matrix)
    pca1.fit(normalized_train_matrix)
    reduced_train_matrix = pca1.transform(normalized_train_matrix)
    print("Number of features before reduction", len(train_matrix[0]))
    print("Number of features after reduction", len(reduced_train_matrix[0]))
    
    # Train Neural network
    print("Training Neural Network on reduced feature matrix")
    model3 = NeuralNetworkClassifier(reduced_train_matrix, train_labels, 1000)
    model3.trainNeuralNetwork()
    
    print("Predict with SVM and Naive Bayes")
    # Predict with SVM and Naive Bayes
    result1 = model1.predict(test_matrix)
    result2 = model2.predict(test_matrix)
    
    # Predict with Neural Network
    pca2 = PCA(len(reduced_train_matrix[0]))
    normalized_test_matrix = StandardScaler().fit_transform(test_matrix)
    pca2.fit(normalized_test_matrix)
    reduced_test_matrix = pca2.transform(normalized_test_matrix)
    print("Predict with Neural Network")
    result3 = model3.predict(test_matrix)
    
    # Result
    print("Confusion matrix for SVM")
    print(confusion_matrix(test_labels, result1))
    print("Confusion matrix for Naive Bayes")
    print(confusion_matrix(test_labels, result2))
    print("Confusion matrix for Neural network")
    print(confusion_matrix(test_labels, result3))
    pass
