import os
from collections import Counter
import numpy as np
import csv


# write it into CSV
def writeMatrixToFile(matrix, fileName):
    with open(fileName, 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in matrix]
    
# read CSV and return a numpy array
def readMatrixFromFile(fileName):
    return np.genfromtxt(fileName, delimiter=',')

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                # Dataset's format is such that email's content starts from the third line
                if i == 2:
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary
    
def extract_features(mail_dir, dictionary):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 3000))
    docID = 0;
    for fil in files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                # Dataset's format is such that email's content starts from the third line
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                print("[",docID,",",wordID,"] = ", words.count(word))
                                features_matrix[docID, wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix