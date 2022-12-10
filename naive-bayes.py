from sklearn.model_selection import train_test_split
import numpy as np
from preprocessing import *
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.ensemble import RandomForestClassifier
# def separate_by_class(data):
#     classes = {}
#     for row in data:
#         if row[0] not in classes:
#             classes[row[0]] = []
#         classes[row[0]].append(row)
#     return classes

# dataset = [[0,3.393533211,2.331273381],
# 	[0,3.110073483,1.781539638],
# 	[0,1.343808831,3.368360954],
# 	[0,3.582294042,4.67917911],
# 	[0,2.280362439,2.866990263],
# 	[1,7.423436942,4.696522875],
# 	[1,5.745051997,3.533989803],
# 	[1,9.172168622,2.511101045],
# 	[1,7.792783481,3.424088941],
# 	[1,7.939820817,0.791637231]]

# separated = separate_by_class(dataset)
# for label in separated:
# 	print(label)
# 	for row in separated[label]:
# 		print(row)

if __name__ == '__main__':
    data = convert_np()
    Y = data[:, 0]
    X = data[:, 1:]
    Xtrn, Xtst, Ytrn, Ytst = train_test_split(X, Y, test_size=0.10, random_state=42)

    model = GaussianNB()
    model.fit(Xtrn, Ytrn)
    print(model.score(Xtst, Ytst))

    # clf = RandomForestClassifier(max_depth=1000, random_state=42)
    # clf.fit(Xtrn, Ytrn)
    # print(clf.score(Xtst, Ytst))