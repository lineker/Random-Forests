import treepredict 
import treerandom
import big_treerandom
import kcrossvalidation
from exampleentry import *
import sys
import data_handling as dt
from prog_bar import ProgBar
import random
from math import sqrt
import argparse

def get_file(filename):
    """
    Tries to extract a filename from the command line.  If none is present, it
    prompts the user for a filename and tries to open the file.  If the file 
    exists, it returns it, otherwise it prints an error message and ends
    execution. 
    """

    try:
        fin = open(filename, "r")
    except IOError:
        print "Error: The file '%s' was not found on this system." % filename
        sys.exit(0)

    return fin

def classify_output(classifier, base, k = -1):
    print "Classifying testx.txt --> result.csv"
    fin = open("testx.txt", "r")
    pb = ProgBar()
    lines = dt.get_lines(fin,float," ", callback = pb.callback)
    del pb
    testdata = dt.transform_features(lines)
    name = "result.csv"
    if k > -1:
        name = "result"+str(k)+".csv"

    resultset = open(name,"w")
    for example in testdata:
        #print len(example)
        result = base.classify(example,classifier)
        resultset.write(str(result)+'\n')
    fin.close()
    resultset.close()

def get_training_validation_set(fin,finy,training_start,training_end,validation_start,validation_end):
    labels = dt.get_lines(finy,int)

    pb = ProgBar()
    lines = dt.get_lines(fin,float," ", callback = pb.callback)
    del pb
    
    #selecting training set
    training_features = dt.select_subset(lines,start=training_start,end=training_end)
    training_labels = dt.select_subset(labels,start=training_start,end=training_end)

    #normalizing features
    training_features = dt.transform_features(training_features)
    training_data = dt.add_labels_to_lines(training_features, labels)

    #selecting validation set
    validation_features = dt.select_subset(lines,start=validation_start,end=validation_end)
    validation_labels = dt.select_subset(labels,start=validation_start,end=validation_end)

    #handling features
    validation_features = dt.transform_features(validation_features)
    validation_data = [ exampleentry(validation_features[i],validation_labels[i]) for i in range(0,len(validation_features)) ]
    
    #random.shuffle(training_data)

    return (training_data,validation_data)   

def train_simple_tree(training_data):
    print "Training Simple Tree"
    tree = treepredict.buildtree(training_data)
    return tree
def train_randomized_forest(training_data):
    print "Training Random Forest"
    pb = ProgBar()
    #m=100,kcandidates=10,nmin=15 -> 53%
    forest = treerandom.build_randomized_forest(training_data,m=100,kcandidates=5,nmin=5, callback = pb.callback) 
    del pb
    return forest
def train_big_randomized_forest(training_data):
    print "Training Random Big Forest of Forest"
    pb = ProgBar()
    forest = big_treerandom.build_random_big_forest(training_data,m=100,kcandidates=5,nmin=5,number_of_forests=10, callback = pb.callback) 
    del pb
    return forest

def accuracy(test_data, classifier, base):
    print "Calculating accuracy"
    corrects = 0
    #classify a set of entries
    for example in test_data:
        #print example.features
        result = base.classify(example.features,classifier)

        if type(result) is dict:
            print str(result.keys()[0]) + "-->" + str(example.label)
            if(result.keys()[0] == example.label):
                corrects = corrects + 1
        else:
            if(result == example.label):
                corrects = corrects + 1

    #calculate the % of accuracy
    print "accuracy = " + str((corrects*100)/len(test_data)) + "%"
    return float(corrects)/float(len(test_data))

def run_k_times(k=1):
    for i in range(0,k):

        fin = get_file("trainx.txt")
        finy = get_file("trainy.csv")

        kcrossvalidation.do_kcross_validation(fin,finy,10)

        #kcrossvalidation.do_simpletree_kcross_validation(fin,finy,5)

        #(training,validation) = get_training_validation_set(fin,finy,training_start=0,training_end=2500,validation_start=2000,validation_end=2500)

        #tree = train_simple_tree(training)

        #treepredict.prune(tree,1)

        #print "accuracy : " + str(accuracy(validation,tree, treepredict))

        #forest = train_randomized_forest(training)
        #print "accuracy : " + str(accuracy(validation,forest, treerandom))

        #big_forest = train_big_randomized_forest(training)
        #print "accuracy : " + str(accuracy(validation,big_forest, big_treerandom))

        #classify_output(forest, treerandom)
        #classify_output(forest, big_treerandom)

        fin.close()
        finy.close()
def command_line():
    args = parser.parse_args()

    if(args.validation):
        if(args.kfolds):
            fin = get_file(args.trainx[0])
            finy = get_file(args.trainy[0])
            if(args.method and args.method[0]=='forest'):
                kcrossvalidation.do_kcross_validation(fin,finy,int(args.kfolds[0]))           
           
    if(args.testx):
        fin = get_file(args.trainx[0])
        finy = get_file(args.trainy[0])
        (training,validation) = get_training_validation_set(fin,finy,training_start=0,training_end=2500,validation_start=0,validation_end=1)
        if(args.method and args.method[0]=='forest'):
            forest = train_randomized_forest(training)
            classify_output(forest, treerandom)
            #print "accuracy : " + str(accuracy(validation,forest, treerandom))
            print "done"


parser = argparse.ArgumentParser(description='Classify Genre of songs using Decision Tree & Random Forest Classifiers')
parser.add_argument('--trainx', '-x', metavar='X', type=str, nargs='+',help='training set without labels', required=True)
parser.add_argument('--trainy','-y', metavar='Y', type=str, nargs='+', help='training set labels', required=True)
parser.add_argument('--testx','-t', metavar='T', type=str, nargs='+', help='training set labels')
parser.add_argument('--validation','-v', metavar='V', type=str, nargs='+', help='activate k cross validation')
parser.add_argument('--kfolds','-k', metavar='K', type=str, nargs='+', help='choose number of folds for cross validation')
parser.add_argument('--method','-m', metavar='M', type=str, nargs='+', help='choose method random forest or decision tree',required=True)
parser.add_argument('--prune','-p', metavar='P', type=str, nargs='+', help='activate pruning for simple decision tree')

if __name__ == "__main__":
    command_line()
    #run_k_times(1)
    