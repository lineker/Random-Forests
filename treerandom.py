from decisionnode import *
import random
import treepredict

# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows,column,value):
   # Make a function that tells us if a row is in 
   # the first group (true) or the second group (false)
   split_function=None
   if isinstance(value,int) or isinstance(value,float):
      split_function=lambda row:row[column]>=value
   else:
      split_function=lambda row:row[column]==value
   
   # Divide the rows into two sets and return them
   set1=[row for row in rows if split_function(row)]
   set2=[row for row in rows if not split_function(row)]
   return (set1,set2)

# Entropy is the sum of p(x)log(p(x)) across all 
# the different possible results
def entropy(rows):
  from math import log
  log2=lambda x:log(x)/log(2)  
  results=uniquecounts(rows)
  # Now calculate the entropy
  ent=0.0
  for r in results.keys():
    p=float(results[r])/len(rows)
    ent=ent-p*log2(p)

  #print ent
  return ent

# Create counts of possible results (the last column of 
# each row is the result)
# unique counts based on the label
#eg. {'1': 152, '0': 168, '3': 177, '2': 185, '4': 59}
def uniquecounts(rows):
  #print len(rows)
  #pdb.set_trace()
  results={}
  for row in rows:
    # The result is the last column
    r=row[len(row)-1]
    #print len(row)
    if r not in results: results[r]=0
    results[r]+=1
  #print "unicounts size : " + str(len(results)) + "size rows:" + str(len(rows))
  #print results
  return results

#choose median as cutting point
def get_cutting_point(column_values):
	values = column_values.keys()
	values.sort()
	#print values
	return values[len(values)/2]

# Probability that a randomly placed item will
# be in the wrong category
def giniimpurity(rows):
  total=len(rows)
  counts=uniquecounts(rows)
  imp=0
  for k1 in counts:
    p1=float(counts[k1])/total
    for k2 in counts:
      if k1==k2: continue
      p2=float(counts[k2])/total
      imp+=p1*p2
  return imp

def pick_candidate_random(candidates, rows):
	return random.choice(candidates)

def pick_candidate_entropy(candidates, rows):
  #use method entropy to choose best candidate 
  current_score = entropy(rows)
  best_candidate = random.choice(candidates)
  best_gain=0.0

  for candidate in candidates:
    col = candidate[0]
    value = candidate[1]
    #split set based on the feature and value
    (set1,set2)=divideset(rows,col,value)

    # Information gain
    p=float(len(set1))/len(rows)
    gain=current_score-p*entropy(set1)-(1-p)*entropy(set2)
    if gain>best_gain and len(set1)>0 and len(set2)>0:
      best_gain=gain
      best_candidate = candidate
  #print best_candidate
  return best_candidate

def pick_candidate_gini(candidates,rows):
  current_score = giniimpurity(rows)
  best_candidate = random.choice(candidates)
  best_gain=0.0

  for candidate in candidates:
    col = candidate[0]
    value = candidate[1]
    #split set based on the feature and value
    (set1,set2)=divideset(rows,col,value)
    #print "current gini= "+str(current_score)
    #print "gini1="+str(giniimpurity(set1)) + "  gini2="+str(giniimpurity(set2))
    # Information gain
    p=float(len(set1))/len(rows)
    gain=current_score-p*giniimpurity(set1)-(1-p)*giniimpurity(set2)
    if gain>best_gain and len(set1)>0 and len(set2)>0:
      best_gain=gain
      best_candidate = candidate
  #print best_candidate
  return best_candidate

def pick_candidate_gini_overall(candidates,rows):
  current_score = giniimpurity(rows)
  best_candidate = random.choice(candidates)
  best_gain=0.0

  for candidate in candidates:
    col = candidate[0]
    for value in candidate[1]:
      #value = candidate[1]
      #split set based on the feature and value
      (set1,set2)=divideset(rows,col,value)
      #print "current gini= "+str(current_score)
      #print "gini1="+str(giniimpurity(set1)) + "  gini2="+str(giniimpurity(set2))
      # Information gain
      p=float(len(set1))/len(rows)
      gain=current_score-p*giniimpurity(set1)-(1-p)*giniimpurity(set2)
      if gain>best_gain and len(set1)>0 and len(set2)>0:
        best_gain=gain
        best_candidate = candidate
  #print best_candidate
  return best_candidate


def get_voting_result(rows):
  winner_key = 0
  winner_value = 0
  pickrandom = False
  counts = uniquecounts(rows)
	
  for key in counts.keys():
    if(counts[key] > winner_value):
      winner_key = key
      winner_value = counts[key]
    elif(counts[key] == winner_value):
      pickrandom = True

  if(pickrandom):
    winner_key = random.choice(counts.keys())

  dic = {winner_key:winner_value}

  return dic

#rows = training data
#kcandidates = number of candidates to pick
#nmin number of examples a node need to have.
#scoref = function used to choose candidates
def buildrandomtree_gini_overall(rows,kcandidates,nmin,pickcandidate=pick_candidate_gini):
  rows = rows[:]
  if len(rows)==0: return decisionnode()

  candidates = []
  
  column_count=len(rows[0])-1
  #print "number of columns = " + str(column_count)
  #pick k random candidates
  #candidate = (column_index,value)
  for i in range(0,kcandidates):
  	random_index = random.randint(0,column_count-1)
  	
  	#get all unique values for a specific feature (column)
  	column_values={}
  	for row in rows:
  		column_values[row[random_index]]=1

  	#get a cutting point
  	cutting_point = column_values.keys()
  	#print "rand feature index ="+str(random_index)+ "\n cutting point="+str(cutting_point)
  	#add to list of candidates
  	candidates.append((random_index,cutting_point))

  #print candidates
  #choose a candidate based on function given
  chosen_candidate = pick_candidate_gini_overall(candidates,rows)

  #print chosen_candidate

  col = chosen_candidate[0]
  value = chosen_candidate[1]
  #split set based on the feature and value
  (set1,set2)=divideset(rows,col,value)

  #set1 = truebranch
  trueBranch = None
  #set2 = falsebranch
  falseBranch = None
  #print "item in leaf1 = " +str(len(set1))
  #check if set1 has the min size
  if(len(set1)<=nmin):
  	#do voting on the elements of of set1
  	#set and answer for this true branch
  	voting_result = get_voting_result(set1)
  	trueBranch = decisionnode(results=voting_result)
  else:
  	#it means we need to grow this 
  	trueBranch = buildrandomtree(set1,kcandidates,nmin,pickcandidate)
  #print "item in leaf2 = " +str(len(set2))
  #check if set2 has the min size
  if(len(set2)<=nmin):
  	#do voting on the elements of of set2
  	#set and answer for this true branch
  	voting_result = get_voting_result(set2)
    #uniquecounts
  	falseBranch = decisionnode(results=voting_result)
  else:
  	#it means we need to grow this 
  	falseBranch = buildrandomtree(set2,kcandidates,nmin,pickcandidate)

  return decisionnode(col=col,value=value,tb=trueBranch,fb=falseBranch)

#rows = training data
#kcandidates = number of candidates to pick
#nmin number of examples a node need to have.
#scoref = function used to choose candidates
def buildrandomtree(rows,kcandidates,nmin,pickcandidate=pick_candidate_gini):
  rows = rows[:]
  if len(rows)==0: return decisionnode()

  candidates = []
  
  column_count=len(rows[0])-1
  #print "number of columns = " + str(column_count)
  #pick k random candidates
  #candidate = (column_index,value)
  for i in range(0,kcandidates):
    random_index = random.randint(0,column_count-1)
    
    #get all unique values for a specific feature (column)
    column_values={}
    for row in rows:
      column_values[row[random_index]]=1

    #get a cutting point
    cutting_point = get_cutting_point(column_values)
    #print "rand feature index ="+str(random_index)+ "\n cutting point="+str(cutting_point)
    #add to list of candidates
    candidates.append((random_index,cutting_point))

  #print candidates
  #choose a candidate based on function given
  chosen_candidate = pickcandidate(candidates,rows)

  #print chosen_candidate

  col = chosen_candidate[0]
  value = chosen_candidate[1]
  #split set based on the feature and value
  (set1,set2)=divideset(rows,col,value)

  #set1 = truebranch
  trueBranch = None
  #set2 = falsebranch
  falseBranch = None
  #print "item in leaf1 = " +str(len(set1))
  #check if set1 has the min size
  if(len(set1)<=nmin):
    #do voting on the elements of of set1
    #set and answer for this true branch
    voting_result = get_voting_result(set1)
    trueBranch = decisionnode(results=voting_result)
  else:
    #it means we need to grow this 
    trueBranch = buildrandomtree(set1,kcandidates,nmin,pickcandidate)
  #print "item in leaf2 = " +str(len(set2))
  #check if set2 has the min size
  if(len(set2)<=nmin):
    #do voting on the elements of of set2
    #set and answer for this true branch
    voting_result = get_voting_result(set2)
    #uniquecounts
    falseBranch = decisionnode(results=voting_result)
  else:
    #it means we need to grow this 
    falseBranch = buildrandomtree(set2,kcandidates,nmin,pickcandidate)

  return decisionnode(col=col,value=value,tb=trueBranch,fb=falseBranch)

def printtree(tree,indent=''):
   # Is this a leaf node?
   if tree.results!=None:
      print str(tree.results)
   else:
      # Print the criteria
      print 'row['+str(tree.col)+']>='+str(tree.value)+'? '

      # Print the branches
      print indent+'T->',
      printtree(tree.tb,indent+'  ')
      print indent+'F->',
      printtree(tree.fb,indent+'  ')

"""

RANDOMIZED FOREST START HERE

"""
#this version receives a subset of features
def build_randomized_forest(rows,m,kcandidates,nmin,pickcandidate=pick_candidate_gini,callback=None):
  forest = []
  for i in range(0,m):
    #tree = buildrandomtree(rows,kcandidates,nmin,pickcandidate=pickcandidate)
    tree = buildrandomtree(rows,kcandidates,nmin,pickcandidate=pickcandidate)
    forest.append(tree)
    if callback: callback(i,m-i)
  #print "size of forest = " + str(len(forest))

  #print "building using pick random"
  #for i in range(0,m/2):
  #  tree = buildrandomtree(rows,kcandidates,nmin,pickcandidate=pick_candidate_random)
  #  forest.append(tree)
  #  if callback: callback(i,m-i)

  return forest

def classify(example,forest):
  #print "using forest to classify"

  counts = {}

  #count the results
  for i in range(0,len(forest)):
    #printtree(forest[0])
    r = treepredict.classify(example,forest[i])
    if r not in counts: counts[r]=1
    counts[r]+=1

  
  winner_key = 0
  winner_value = 0
  pickrandom = False
  for key in counts.keys():
    if(counts[key] > winner_value):
      winner_key = key
      winner_value = counts[key]
    elif (counts[key] == winner_value):
      pickrandom = True

    if(pickrandom):
      winner_key = random.choice(counts.keys())
  
  #print counts

  return winner_key