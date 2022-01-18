# print('heee')

import json
import jsonlines
import pickle
import six
# from eval import compute_rouge_n,tokenizef,_summary_level_lcs,sentencelevelrougeL

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from rouge_score import rouge_scorer
import statistics
from scipy.stats import kendalltau as correlator
from syntaticeval import weightedrouge

# print('hee')
# f=open('model_annotations.aligned.paired.jsonl')



def convertdec(curr_decoded):
	return curr_decoded.replace('. ',' .\n')

# i=0
# x=0
# coherence=[]
# consistency=[]
# fluency=[]
# relevance=[]
# with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
# 	for obj in reader:

# 		curridx=i
# 		idealidx=''
# 		idealidx=idealidx+str(curridx)
# 		rempos=6-len(idealidx)
# 		for j in range(rempos):
# 		  idealidx='0'+idealidx

# 		print(idealidx)

# 		# # print(obj['expert_annotations'])
# 		# t1=obj['expert_annotations'][0]
# 		# t2=obj['expert_annotations'][1]
# 		# t3=obj['expert_annotations'][2]
# 		# # consistency': 1, 'fluency': 4, 'relevance': 2
# 		# coherenceval=(t1['coherence']+t2['coherence']+t3['coherence'])/3
# 		# consistencyval=(t1['consistency']+t2['consistency']+t3['consistency'])/3
# 		# fluencyval=(t1['fluency']+t2['fluency']+t3['fluency'])/3
# 		# relevanceval=(t1['relevance']+t2['relevance']+t3['relevance'])/3
# 		# coherence.append(coherenceval)
# 		# consistency.append(consistencyval)
# 		# fluency.append(fluencyval)
# 		# relevance.append(relevanceval)

# 		# for k in [2,3,4,5,6,7,8,9,10]:
# 		# 	add_references='reference/'+idealidx+'_reference'+str(k)+'.txt'

# 		# 	curr_reference=obj['references'][k]
# 		# 	decoded=convertdec(curr_reference)
# 		# 	file1 = open(add_references,"w")
# 		# 	file1.write(decoded)

# 		# print(obj['model_id'])

# 		if obj['model_id']=='M8':
# 			t1=obj['expert_annotations'][0]
# 			t2=obj['expert_annotations'][1]
# 			t3=obj['expert_annotations'][2]
# 			# consistency': 1, 'fluency': 4, 'relevance': 2
# 			coherenceval=(t1['coherence']+t2['coherence']+t3['coherence'])/3
# 			consistencyval=(t1['consistency']+t2['consistency']+t3['consistency'])/3
# 			fluencyval=(t1['fluency']+t2['fluency']+t3['fluency'])/3
# 			relevanceval=(t1['relevance']+t2['relevance']+t3['relevance'])/3
# 			coherence.append(coherenceval)
# 			consistency.append(consistencyval)
# 			fluency.append(fluencyval)
# 			relevance.append(relevanceval)
# 			# curridx2=x
# 			# idealidx2=''
# 			# idealidx2=idealidx2+str(curridx2)
# 			# rempos2=6-len(idealidx2)
# 			# for j2 in range(rempos2):
# 			#   idealidx2='0'+idealidx2


# 			# for k in [0,1,2,3,4,5,6,7,8,9,10]:
# 			# 	add_references='M8reference2/'+idealidx2+'_reference'+str(k)+'.txt'

# 			# 	curr_reference=obj['references'][k]
# 			# 	decoded=convertdec(curr_reference)
# 			# 	file1 = open(add_references,"w")
# 			# 	file1.write(decoded)


# 			# curr_decoded=obj['decoded']
# 			# decoded=convertdec(curr_decoded)
# 			# add_decoded='M8decoded/'+idealidx2+'_decoded.txt'
# 			# file1=open(add_decoded,'w')
# 			# file1.write(decoded)
# 			# x=x+1


		
# 		i=i+1


# # print(coherence)
# # print(consistency)
# # print(fluency)
# # print(relevance)





# with open("coherenceM8.txt", "wb") as fp:
# 	pickle.dump(coherence, fp)

# with open("consistencyM8.txt", "wb") as fp:
# 	pickle.dump(consistency, fp)

# with open("fluencyM8.txt", "wb") as fp:
# 	pickle.dump(fluency, fp)

# with open("relevanceM8.txt", "wb") as fp:
# 	pickle.dump(relevance, fp)
# with open("coherence.txt", "rb") as fp:
# 	b = pickle.load(fp)


# print(b)

# for k in range(len(coherence)):
# 	if coherence[k]!=b[k]:
# 		print('False')

# print('true')



temp='M'
modelID=[]
for i in [0,1,2,5,8,9,10,11,12,13,14,15,17,20,22,23]:
	temp2=temp+str(i)
	# print(temp2)
	modelID.append(temp2)
# modelID.append('M23_dynamicmix')
rougelistmodelwise=[]
coherencelistmodelwise=[]
consistencylistmodelwise=[]
fluencylistmodelwise=[]
relevancelistmodelwise=[]
originallistmodelwise=[]

iq=0
for m in ['M8']:
	print(m)


	rougelist=[]
	coherencelist=[]
	consistencylist=[]
	fluencylist=[]
	relevancelist=[]
	with jsonlines.open('model_annotations.aligned.paired.jsonl') as reader:
		for obj in reader:
			if 1==1:
				print(iq)
				iq=iq+1
				decoded=obj['decoded']
				t1=obj['expert_annotations'][0]
				t2=obj['expert_annotations'][1]
				t3=obj['expert_annotations'][2]
				# consistency': 1, 'fluency': 4, 'relevance': 2
				coherenceval=(t1['coherence']+t2['coherence']+t3['coherence'])/3
				consistencyval=(t1['consistency']+t2['consistency']+t3['consistency'])/3
				fluencyval=(t1['fluency']+t2['fluency']+t3['fluency'])/3
				relevanceval=(t1['relevance']+t2['relevance']+t3['relevance'])/3
				coherencelist.append(coherenceval)
				consistencylist.append(consistencyval)
				fluencylist.append(fluencyval)
				relevancelist.append(relevanceval)

				currrouge=0
				for k in [0,1,2,3,4,5,6,7,8,9,10]:
					ref=obj['references'][k]
					article=obj['text']
					currrougetemp=weightedrouge(ref,decoded,article,2)

					# print(decoded)

					# predicted_summ_sent = six.ensure_str(decoded).split(". ")
					# gold_summ_sent= six.ensure_str(ref).split(". ")

					# gold_summ_sent2=[]
					# for line in gold_summ_sent:
					#   if len(line)!=0:
					#     gold_summ_sent2.append(line)

					# predicted_summ_sent2=[]
					# for line in predicted_summ_sent:
					#   if len(line)!=0:
					#     predicted_summ_sent2.append(line)

					# print(predicted_summ_sent2)
					# print(gold_summ_sent2)

					# q,w,currrougetemp=sentencelevelrougeL(ref,decoded)


					# currrougetemp=compute_rouge_n(tokenizef(decoded,True),tokenizef(ref,True),2,'f')

					# scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
					# scores2 = scorer.score(ref,
					#                       decoded)

					# currrougetemp=scores2['rouge1'][2]

					# currrouge=max(currrougetemp,currrouge)
					currrouge+=currrougetemp
				currrouge=currrouge/11
				rougelist.append(currrouge)
			else:
				continue

		if len(rougelist)!=0:
			rougelistmodelwise.append(statistics.mean(rougelist))
			print(rougelistmodelwise)
			coherencelistmodelwise.append(statistics.mean(coherencelist))
			# print(coherencelistmodelwise)
			consistencylistmodelwise.append(statistics.mean(consistencylist))
			# print(consistencylistmodelwise)
			fluencylistmodelwise.append(statistics.mean(fluencylist))
			# print(fluencylistmodelwise)
			relevancelistmodelwise.append(statistics.mean(relevancelist))
			# print(relevancelistmodelwise)


	print('============================')
	originallist=[]
	with jsonlines.open('model_annotations.aligned.scored.jsonl') as reader:
		for obj in reader:
			if obj['model_id']!='M23_dynamicmix':
				originallist.append(obj['metric_scores_11']['rouge']['rouge_2_f_score'])

		if len(originallist)!=0:
			originallistmodelwise.append(statistics.mean(originallist))



	print(kendalltau(originallist,rougelist))
	print(kendalltau(rougelist,coherencelist))
	print(kendalltau(originallist,coherencelist))
	print(kendalltau(rougelist,consistencylist))
	print(kendalltau(originallist,consistencylist))
	print(kendalltau(rougelist,fluencylist))
	print(kendalltau(originallist,fluencylist))
	print(kendalltau(rougelist,relevancelist))
	print(kendalltau(originallist,relevancelist))


	










# count=0
# for i in range(len(rougelist)):
# 	if originallist[i]!=rougelist[i]:
# 		count+=1
# 		print('there f score: {} and our f score: {}'.format(originallist[i],rougelist[i]))



# print(count)
# print(statistics.mean(rougelist))
# print(statistics.mean(originallist))




# corr4, _ = kendalltau(coherencelistmodelwise, originallistmodelwise)
# print('Kendall Rank correlation with coherence: {:.3f}'.format(corr4))

# print(statistics.mean(originallistmodelwise))
# print(len(originallistmodelwise))

# print(len(rougelistmodelwise))
corr1, _ = kendalltau(coherencelistmodelwise, rougelistmodelwise)
print('Kendall Rank correlation with coherence and our implemented rouge 1: {:.3f}'.format(corr1))
corr1o=correlator(coherencelistmodelwise, originallistmodelwise)[0]
print('Kendall Rank correlation with coherence and their implemented rouge 1: {:.3f}'.format(corr1o))


corr2, _ = kendalltau(consistencylistmodelwise, rougelistmodelwise)
print('Kendall Rank correlation with consistency and our implemented rouge 1: {:.3f}'.format(corr2))
corr2o=correlator(consistencylistmodelwise, originallistmodelwise)[0]
print('Kendall Rank correlation with consistency and theri implemented rouge 1: {:.3f}'.format(corr2o))


corr3, _ = kendalltau(fluencylistmodelwise, rougelistmodelwise)
print('Kendall Rank correlation with fuency and our implemented rouge 1: {:.3f}'.format(corr3))
corr3o=correlator(fluencylistmodelwise, originallistmodelwise)[0]
print('Kendall Rank correlation with fuency and their implemented rouge 1: {:.3f}'.format(corr3o))


corr4, _ = kendalltau(relevancelistmodelwise, rougelistmodelwise)
print('Kendall Rank correlation with relevance and our implemented rouge 1: {:.3f}'.format(corr4))
corr4o=correlator(relevancelistmodelwise, originallistmodelwise)[0]
print('Kendall Rank correlation with relevance and their implemented rouge 1: {:.3f}'.format(corr4o))


# file1=open('result1wc2.txt','w')
# file1.write('Kendall Rank correlation with coherence and our implemented weighted rouge 1: {:.3f}'.format(corr1))
# file1.write('\n')
# file1.write('Kendall Rank correlation with coherence and their implemented weighted rouge 1: {:.3f}'.format(corr1o))
# file1.write('\n')
# file1.write('Kendall Rank correlation with consistency and our implemented weighted rouge 1: {:.3f}'.format(corr2))
# file1.write('\n')
# file1.write('Kendall Rank correlation with consistency and theri implemented weighted rouge 1: {:.3f}'.format(corr2o))
# file1.write('\n')
# file1.write('Kendall Rank correlation with fuency and our implemented weighted rouge 1: {:.3f}'.format(corr3))
# file1.write('\n')
# file1.write('Kendall Rank correlation with fuency and their implemented weighted rouge 1: {:.3f}'.format(corr3o))
# file1.write('\n')
# file1.write('Kendall Rank correlation with relevance and our implemented weighted rouge 1: {:.3f}'.format(corr4))
# file1.write('\n')
# file1.write('Kendall Rank correlation with relevance and their implemented weighted rouge 1: {:.3f}'.format(corr4o))
# file1.write('\n')




# print(kendalltau(originallistmodelwise,rougelistmodelwise))


