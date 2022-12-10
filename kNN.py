from preprocessing import *
from sklearn.model_selection import train_test_split
from numpy import dot
from numpy.linalg import norm
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
 
def most_frequent(list):
    occurence_count = Counter(list)
    return occurence_count.most_common(1)[0][0]

if __name__ == '__main__':
    data = convert_np()
    Y = data[:, 0]
    X = data[:, 1:]
    Xtrn, Xtst, Ytrn, Ytst = train_test_split(X, Y, test_size=0.10, random_state=42)

    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(Xtrn, Ytrn)
    print(model.score(Xtrn, Ytrn))
    # k = 15
    # res = 0
    # for i in range(len(Xtst)):
    #     sims = []
    #     for j in range(len(Xtrn)):
    #         sim = dot(Xtst[i], Xtrn[j])/(norm(Xtst[i])*norm(Xtrn[j]))
    #         sims.append((sim, Ytrn[j]))
    #     sims.sort(key=lambda t:t[0])
    #     sims = sims[-(k-1):]
    #     sims = [x[1] for x in sims]
    #     max_occur = most_frequent(sims)
    #     if max_occur == Ytst[i]:
    #         res += 1
        #print(str(Ytst[i]) + ',' + str(max_occur) + ',' + str(sum(sims)/len(sims)))

    # print(res/len(Ytst))


