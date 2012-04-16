from prog_bar import ProgBar
from math import sqrt

def average(values):
    return sum(values) / len(values)

def stdDeviation(variance):
	if(variance > 0):
		return sqrt(variance) 
	return variance

def variance(data):
  if len(rows)==0: return 0
  mean = average(data)
  variance=sum([(d-mean)**2 for d in data])/len(data)
  return variance

def get_lines(fin,attrtype,separator=None, callback = None):
	print "loading file lines"
	result = []

	lines = [line.strip() for line in fin.readlines()]
	print separator
	if separator != None:
		i = 1
		for line in lines:
			track = [attrtype(datum.strip()) for datum in line.split(separator)]
			result.append(track)
			if callback and len(lines)-i > 0: callback(i,len(lines)-i)
			i = i+1
				
	else:
		result = [attrtype(line.strip()) for line in lines]
	return result

def add_labels_to_lines(lines,labels):
	for i in range(0,len(lines)):
		lines[i].append(labels[i])
	return lines

#end is exclusive , start is inclusive
def select_subset(data,start,end):
	return data[start:end]

def chunks(l, n):
	""" Yield successive n-sized chunks from l.
	"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

def normalize_volume(listsegments):
    allVolumes = [float(row[0]) for row in listsegments]
    mean = average(allVolumes)
    return round(mean,3)

def normalize_timbre(listsegments):
	allTimbres = []
	variances = []
	#Summing pitch feature (12 numbers) for each row (segment)
	for j in xrange(13,25):
		alltimbresForOneSeg = []
		for row in listsegments:
			alltimbresForOneSeg.append(row[j])
		#we get the average of the pitch summation
		avg = average(alltimbresForOneSeg)
		allTimbres.append(round(avg,3))  

		#calc variance for each pitch feature
		#variance=sum([(d-avg)**2 for d in alltimbresForOneSeg])/len(alltimbresForOneSeg)
		#variances.append(variance)
		#allTimbres = allTimbres + [average(variances)] 

	return allTimbres

def normalize_pitch(listsegments):
	#print "normalizing pitch #seg = " + str(len(listsegments))
	allPitchs = []
	variances = []
    #Summing pitch feature (12 numbers) for each row (segment)
	for j in xrange(1,13):
		allpitchsForOneSeg = []
		for row in listsegments:
			allpitchsForOneSeg.append(row[j])
		
		#we get the average of the pitch summation
		avg = average(allpitchsForOneSeg)
		allPitchs.append(round(avg,3))
		
		#calc variance for each pitch feature
		#variance=sum([(d-avg)**2 for d in allpitchsForOneSeg])/len(allpitchsForOneSeg)
		#variances.append(variance)
		#allPitchs = allPitchs + [average(variances)]

		#calc std deviation
		#allPitchs = allPitchs + [average([stdDeviation(var) for var in variances])]

	return allPitchs
def get_chunks(listsegments,k=40):
	segmentsToBeCombined = k
	result = []
	number_chunks = len(listsegments) / segmentsToBeCombined
	#print "will have " + str(number_chunks) + " * 25 = " + str(number_chunks * 25)
	for i in range(0,number_chunks):
		start = i * segmentsToBeCombined
		end = start + segmentsToBeCombined
		chuckof4segs = listsegments[start:end]
		result.append(chuckof4segs)
	#print len(result)
	return result
#We are selection last 25 features + 25 features in the middle + 25 avg features
def features_selection_25avg(listsegments):
	avg25 = [normalize_volume(listsegments)] + normalize_pitch(listsegments) + normalize_timbre(listsegments)
	return avg25 + listsegments[len(listsegments)/2] + listsegments[-1]

#it averages chunks of k segments
def feature_selection_avg_chunks_of_k_segments(listsegments,k=40):
	segmentsToBeCombined = k
	result = []
	number_chunks = len(listsegments) / segmentsToBeCombined
	#print "will have " + str(number_chunks) + " * 25 = " + str(number_chunks * 25)
	for i in range(0,number_chunks):
		start = i * segmentsToBeCombined
		end = start + segmentsToBeCombined
		chuckof4segs = listsegments[start:end]
		avgchunks = [normalize_volume(chuckof4segs)] + normalize_pitch(chuckof4segs) + normalize_timbre(chuckof4segs)
		result = result + avgchunks
	#print len(result)
	return result

#averaging each segment inside each other [3] * 200 
def feature_selection_averaging_segment_focus(listsegments):
	result = []
	for segment in listsegments:
		volume = segment[0]
		avgpitchs = average(segment[1:13])
		avgtimbres = average(segment[13:25])
		result = result + [volume,avgpitchs,avgtimbres]
	return result

#avg features vertically, end up 1 segment of 25 features
def feature_selection_averaging_feature(listsegments):

    result = [normalize_volume(listsegments)] + normalize_pitch(listsegments) + normalize_timbre(listsegments)
    return result

def feature_selection_variance_segment_focus(listsegments):
	result = []
	for segment in listsegments:
		volume = segment[0]
		pitchs = segment[1:13]
		avgpitchs = average(pitchs)
		variance_pitch=sum([(d-avgpitchs)**2 for d in pitchs])/len(pitchs)
		timbres = segment[13:25]
		avgtimbres = average(timbres)
		variance_timbres=sum([(d-avgtimbres)**2 for d in timbres])/len(timbres)
		result = result + [volume,variance_pitch,variance_timbres]
		#result = result + [volume,avgpitchs,avgtimbres,variance_pitch,variance_timbres]
	return result

def feature_selection_variance_per_feature(listsegments):

	#volumen variance
	allVolumes = [float(row[0]) for row in listsegments]
	mean_volume = average(allVolumes)
	variance_volume = sum([(d-mean_volume)**2 for d in allVolumes])/len(allVolumes)

	t_variances = []
	#Summing pitch feature (12 numbers) for each row (segment)
	for j in xrange(13,25):
		alltimbresForOneSeg = []
		for row in listsegments:
			alltimbresForOneSeg.append(row[j])
		#we get the average of the pitch summation
		avg = average(alltimbresForOneSeg)

		#calc variance for each pitch feature
		variance=sum([(d-avg)**2 for d in alltimbresForOneSeg])/len(alltimbresForOneSeg)
		t_variances.append(variance) 

	p_variances = []
	#Summing pitch feature (12 numbers) for each row (segment)
	for j in xrange(1,13):
		allpitchsForOneSeg = []
		for row in listsegments:
			allpitchsForOneSeg.append(row[j])

		#we get the average of the pitch summation
		avg = average(allpitchsForOneSeg)

		#calc variance for each pitch feature
		variance=sum([(d-avg)**2 for d in allpitchsForOneSeg])/len(allpitchsForOneSeg)
		p_variances.append(variance)

	result = [variance_volume] + p_variances + t_variances
	return result

#volume is not calculated variance in this case
def feature_selection_stdDeviation_per_segment(listsegments):
	list_variance = feature_selection_variance_segment_focus(listsegments)
	#print list_variance
	return [stdDeviation(variance) for variance in list_variance]

def feature_selection_stdDeviation_per_feature(listsegments):
	list_variance = feature_selection_variance_per_feature(listsegments)
	return [stdDeviation(variance) for variance in list_variance]

def feature_select_chunks_variance_per_feature(listsegments):
	segments_chunks = get_chunks(listsegments)
	result = []
	for chunk in segments_chunks:
		result = result + feature_selection_variance_per_feature(chunk)
	return result

def feature_select_chunks_stdDeviation_per_feature(listsegments):
	segments_chunks = get_chunks(listsegments)
	result = []
	for chunk in segments_chunks:
		result = result + feature_selection_stdDeviation_per_feature(chunk)
	return result

#it averages chunks of k segments
def feature_selection_avg_chunks_of_k_segments_trimbre_only(listsegments,k=40):
	segmentsToBeCombined = k
	result = []
	number_chunks = len(listsegments) / segmentsToBeCombined
	#print "will have " + str(number_chunks) + " * 25 = " + str(number_chunks * 25)
	for i in range(0,number_chunks):
		start = i * segmentsToBeCombined
		end = start + segmentsToBeCombined
		chuckof4segs = listsegments[start:end]
		avgchunks = normalize_timbre(chuckof4segs)
		result = result + avgchunks
	#print len(result)
	return result

def transform_features(data,feature_selection=feature_selection_avg_chunks_of_k_segments):
	print "Transforming data"
	#split track in 25 segments and do feature selection on them
	tracks = [feature_selection(list(chunks(track,25))) for track in data]
	print str(len(tracks[0])) + " features selected"
	return tracks