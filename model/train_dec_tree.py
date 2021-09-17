import os
import h5py
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
import graphviz



fname = '../data/disruptions.txt'
print 'Reading:', fname
f = open(fname, 'r')

dst = dict()

for pulse in f:
    data = pulse[:-1]
    data = data.split(' ')
    dst[data[0]]=data[1]
f.close()



fname = 'dec_tree_xy.hdf'
print 'Reading:', fname
f = h5py.File(fname, 'r')

inputs = dict()
outputs = dict()
times = dict()

start=-100
end=-50

diff = start-end

for pulse in f:
    g = f[pulse]
    inputs[pulse] = g['input'][:]
    outputs[pulse] = g['output'][:]
    times[pulse] = g['time'][:]

    if pulse in dst:
        i=0
        for item in outputs[pulse]:
            diff = (float(dst[pulse]) - float(times[pulse][i]))
            if diff < 0.200:
                outputs[pulse][i] = 1
            else:
                outputs[pulse][i] = 0
            i+=1


f.close()

X = np.array([[1,1]])
Y = np.array([1])
for pulse in inputs:
    X = np.concatenate((X,np.asarray(inputs[pulse])), axis=0)
    Y = np.concatenate((Y,np.asarray(outputs[pulse])), axis=0)

X = np.delete(X,0,0)
Y = np.delete(Y,0)

print X.shape, Y.shape

X = X.tolist()
Y = Y.tolist()

print len(X), len(Y)

clf = tree.DecisionTreeClassifier(max_depth=4,min_samples_leaf=10)
clf = clf.fit(X, Y)

dot_data = tree.export_graphviz(clf, out_file=None,feature_names=["probability","ttd"],class_names=["non-disruptive","disruptive"],filled=True,rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("disruptions")


print clf.predict([[0.3,10]])
print clf.predict([[0.2,15]])
print clf.predict([[0.4,4]])
print clf.predict([[0.8,1]])
print clf.predict([[0.9,0.5]])
print clf.score(X,Y)

joblib.dump(clf, 'dec_tree.pkl') 
