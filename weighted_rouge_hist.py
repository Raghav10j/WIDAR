import json
import jsonlines
import pickle
import six
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from rouge_score import rouge_scorer
import statistics
from scipy.stats import kendalltau as correlator
from syntaticeval import *

rougerecalllist=[]
rougefscorelist=[]
rougePrecisionlist=[]



iq=0

with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
	for obj in reader:
		# text=obj['text']
		decoded=obj['decoded']
		print(iq)
		iq+=1

		currrougerecall=0
		currrougeprecision=0
		currrougefscore=0
		for k in [0,1,2,3,4,5,6,7,8,9,10]:
			ref=obj['references'][k]
			article=obj['text']
			currrougetemprecall,currrougetempprecision,currrougetempfscore=weightedrouge(ref,decoded,article,2)


			currrougerecall=currrougerecall+currrougetemprecall
			currrougeprecision=currrougeprecision+currrougetempprecision
			currrougefscore=currrougefscore+currrougetempfscore
		currrougerecall=currrougerecall/11
		rougerecalllist.append(currrougerecall)
		currrougeprecision=currrougeprecision/11
		rougePrecisionlist.append(currrougeprecision)
		currrougefscore=currrougefscore/11
		rougefscorelist.append(currrougefscore)


fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(rougerecalllist, bins = [0, 0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1],edgecolor='black')
 
# Show plot
plt.show()
plt.savefig('histograms/Weightedrouge2RecallModified.png')
plt.savefig('histograms/Weightedrouge2RecallModified.pdf')



fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(rougePrecisionlist, bins = [0, 0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1],edgecolor='black')
 
# Show plot
plt.show()
plt.savefig('histograms/weightedrouge2precisionModified.png')
plt.savefig('histograms/weightedrouge2precisionModified.pdf')




fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(rougefscorelist, bins = [0, 0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1],edgecolor='black')
 
# Show plot
plt.show()
plt.savefig('histograms/weightedrouge2fscoreModified.png')
plt.savefig('histograms/weightedrouge2fscoreModified.pdf')