import pickle


def CountFrequency(my_list):
 
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq
 


with open("weightesranges/relevancediffwcov2F1L.txt", "rb") as fp:
	wred = pickle.load(fp)



print(wred)

# rangew=CountFrequency(wred)
# file1 = open("weightesranges/wcovrange.txt","w")
# for key, value in rangew.items():
# 	s=str(key)+' : '+str(value)
# 	file1.write(s)
# 	file1.write('\n')


# Kendall Rank correlation b/w coherence and our final rouge-l fscore: 0.115
# Kendall Rank correlation b/w consistency and our final rouge-l fscore: 0.108
# kenda rank correlation b/w fluency and our final rouge-l fscore: 0.091
# kenda rank correlation b/w relevance and our final rouge-l fscore: 0.218
