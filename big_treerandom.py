import treerandom
import random
from prog_bar import ProgBar

def build_random_big_forest(rows,m,kcandidates,nmin,pickcandidate=treerandom.pick_candidate_random,number_of_forests=10,callback=None):
  forests = []
  for i in range(0,number_of_forests):
    pb = ProgBar()
    forests.append(treerandom.build_randomized_forest(rows,m,kcandidates,nmin,callback=pb.callback))
    del pb
    print i
    if callback: callback(i,number_of_forests-i)
  return forests


def classify(example, big_forest):
  #print "using forest to classify"

  counts = {}

  #count the results
  for i in range(0,len(big_forest)):
    #printtree(forest[0])
    r = treerandom.classify(example,big_forest[i])
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