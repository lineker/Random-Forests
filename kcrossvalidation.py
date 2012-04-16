import data_handling as dt
from prog_bar import ProgBar
from exampleentry import *
import treerandom
import treepredict

def do_kcross_validation(fin,finy,kfolds):
    print "Starting k=" + str(kfolds)+" validation for random forest"
    #there is 2500 tracks
    labels = dt.get_lines(finy,int)
    pb = ProgBar()
    lines = dt.get_lines(fin,float," ", callback = pb.callback)
    del pb
    #normalize features
    
    lines = dt.transform_features(lines)
    data = dt.add_labels_to_lines(lines, labels)


    block_size = len(lines)/kfolds
    print "chunk size = " + str(block_size)
    example_chunks = list(dt.chunks(data, block_size))
    #labels_chunks = list(dt.chunks(labels, block_size))

   
    print "number of chunks = " +str(len(example_chunks))

    #holds avg accuracy for one forest
    accuracy_results = []
    #need to add loop here, to loop over configurations of m,n,k
    m = [100]
    k = [5]
    n = [5]

    bestm = 0
    bestk = 0
    bestn = 0
    bestaccuracy = 0

    for p in range(0,len(m)):
        for f in range(0,len(k)):
            for g in range(0,len(n)):
                for i in range(0,len(example_chunks)):

                    #we leave set in index i out of train
                    print "prepare validation set"
                    validationdata = example_chunks[i]

                    #extract validation chunk
                    print "leaving out block " + str(i) + " for validation"
                    leaveout = i
                    validationdata = [ exampleentry(validationdata[i][0:len(validationdata[i])-1],validationdata[i][-1]) for i in range(0,len(validationdata)) ]
                    
                    trainingdata = []

                    print("merging blocks "),
                    for j in range(0,len(example_chunks)):
                        if(j != leaveout):
                            #print "j="+str(j) + " i="+ str(leaveout)
                            print(str(j) + ","),
                            trainingdata = trainingdata + example_chunks[j]

                    print "\nprepare training set"

                    print "training on " + str(len(trainingdata))
                    print "each track has " + str(len(trainingdata[0])) + " features"
                    pb = ProgBar()
                    forest = treerandom.build_randomized_forest(trainingdata,m=m[p],kcandidates=k[f],nmin=n[g], callback=pb.callback)
                    del pb
                    print "testing on " + str(len(validationdata))
                    corrects = 0
                    #classify a set of entries
                    for example in validationdata:
                        #print example.features
                        result = treerandom.classify(example.features,forest)
                        #print 'expected : ' + str(example.label) + ' result : '+ str(result)
                        if(result == example.label):
                            corrects = corrects + 1
                    #calculate the % of accuracy
                    accuracy_percentage = (corrects*100)/len(validationdata)
                    print "accuracy = " + str(accuracy_percentage) + "%"
                    accuracy_results.append(accuracy_percentage)
                avgcc = dt.average(accuracy_results)
                print "average accuracy using m="+str(m[p]) + ", k="+str(k[f])+", n="+str(n[g]) + "---> " + str(avgcc) + "%"
                if(avgcc > bestaccuracy):
                    bestm = m[p]
                    bestk = k[f]
                    bestn = n[g]
                    bestaccuracy = avgcc
    print "BEST COMBINATION m="+str(bestm) + ", k="+str(bestk)+", n="+str(bestn) + "---> " + str(bestaccuracy) + "%"

def do_simpletree_kcross_validation(fin,finy,kfolds):
    print "Starting k=" + str(kfolds)+" validation for Simple tree"
    #there is 2500 tracks
    labels = dt.get_lines(finy,int)
    pb = ProgBar()
    lines = dt.get_lines(fin,float," ", callback = pb.callback)
    del pb
    #normalize features
    
    lines = dt.transform_features(lines)
    data = dt.add_labels_to_lines(lines, labels)


    block_size = len(lines)/kfolds
    print "chunk size = " + str(block_size)
    example_chunks = list(dt.chunks(data, block_size))
    #labels_chunks = list(dt.chunks(labels, block_size))

   
    print "number of chunks = " +str(len(example_chunks))

    #holds avg accuracy for one forest
    accuracy_results = []

    for i in range(0,len(example_chunks)):

        #we leave set in index i out of train
        print "prepare validation set"
        validationdata = example_chunks[i]

        #extract validation chunk
        print "leaving out block " + str(i) + " for validation"
        leaveout = i
        validationdata = [ exampleentry(validationdata[i][0:len(validationdata[i])-1],validationdata[i][-1]) for i in range(0,len(validationdata)) ]
        
        trainingdata = []

        print("merging blocks "),
        for j in range(0,len(example_chunks)):
            if(j != leaveout):
                #print "j="+str(j) + " i="+ str(leaveout)
                print(str(j) + ","),
                trainingdata = trainingdata + example_chunks[j]

        print "\nprepare training set"

        print "training on " + str(len(trainingdata))
        print "each track has " + str(len(trainingdata[0])) + " features"

        tree = treepredict.buildtree(trainingdata)

        print "testing on " + str(len(validationdata))
        corrects = 0
        #classify a set of entries
        for example in validationdata:
            #print example.features
            result = treepredict.classify(example.features,tree)
            #print 'expected : ' + str(example.label) + ' result : '+ str(result)
            if(result == example.label):
                corrects = corrects + 1
        #calculate the % of accuracy
        accuracy_percentage = (corrects*100)/len(validationdata)
        print "accuracy = " + str(accuracy_percentage) + "%"
        accuracy_results.append(accuracy_percentage)
    avgcc = dt.average(accuracy_results)
    print "average accuracy ="+  str(avgcc) + "%"
