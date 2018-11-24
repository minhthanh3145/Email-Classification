import os
from collections import Counter
import numpy as np
import csv


# write it
def writeMatrixToFile(matrix, fileName):
    with open(fileName, 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in matrix]
    
# read it
def readMatrixFromFile(fileName):
    with open(fileName, 'r') as csvfile:
        reader = csv.reader(csvfile)
        table = [[int(e) for e in r] for r in reader]
        return table

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


def make_Dictionary1(root_dir):
    emails_dirs = [os.path.join(root_dir, f) for f in
                   os.listdir(root_dir)]
    all_words = []
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir, f) for f in
                os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d, f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:
                    for line in m:
                        words = line.split()
                        all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()

    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)

    np.save('dict_enron.npy', dictionary)

    return dictionary


def extract_features1(root_dir, dictionary):
    emails_dirs = [os.path.join(root_dir, f) for f in
                   os.listdir(root_dir)]
    docID = 0
    features_matrix = np.zeros((33716, 3000))
    train_labels = np.zeros(33716)
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir, f) for f in
                os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d, f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:
                    all_words = []
                    for line in m:
                        words = line.split()
                        all_words += words
                    for word in all_words:
                        wordID = 0
                        for (i, d) in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = \
                                    all_words.count(word)
                train_labels[docID] = int(mail.split('.')[-2] == 'spam')
                docID = docID + 1
    return (features_matrix, train_labels)
    
