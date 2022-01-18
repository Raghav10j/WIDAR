import pickle
import matplotlib.pyplot as plt
import numpy as np



with open("weightesranges/coherencediffwcov2F1L.txt", "rb") as fp:
	wredcoh = pickle.load(fp)

with open("weightesranges/consistencydiffwcov2F1L.txt", "rb") as fp:
	wredcon = pickle.load(fp)

with open("weightesranges/fluencydiffwcov2F1L.txt", "rb") as fp:
	wredfl = pickle.load(fp)

with open("weightesranges/relevancediffwcov2F1L.txt", "rb") as fp:
	wredre = pickle.load(fp)





coh=np.array(wredcoh)
con=np.array(wredcon)
fl=np.array(wredfl)
re=np.array(wredre)
coh[0]=0.102
con[0]=0.087
fl[0]=0.06
re[0]=0.204

# y[0]=0.204
# print(y)
x=np.arange(0, 1.1, 0.1)
plt.xticks(x)
plt.plot(x, coh)
plt.plot(x, con,'--')
plt.plot(x, fl,'-.')
plt.plot(x, re,':')
plt.title("Threshold plot for coverage weights",fontdict={'fontsize': 15})

# plt.xlabel('Theta1')
plt.legend(['Coherence','Consistency','Fluency','Relevance'],prop={'size': 12})
# plt.ylabel('correlation scores')
plt.savefig('hyperparameterplots2/wcovthreshold_vals.png')
plt.savefig('hyperparameterplots2/wcovthreshold_vals.pdf')
# Kendall Rank correlation b/w coherence and our final rouge-l fscore: 0.102
# Kendall Rank correlation b/w consistency and our final rouge-l fscore: 0.087
# kenda rank correlation b/w fluency and our final rouge-l fscore: 0.062
# kenda rank correlation b/w relevance and our final rouge-l fscore: 0.204
