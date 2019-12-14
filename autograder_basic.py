import csv
import pickle as pkl


def load_data(ftest, fpred):
    Xtest, Ytest, Ypred = [], [], []

    with open(ftest, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split(' '))
            Xtest.append(rw)

    ftest_label = ftest.split('.')[0] + '_label.csv'
    with open(ftest_label, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = int(row[0])
            Ytest.append(rw)

    with open(fpred, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = int(row[0])
            Ypred.append(rw)

    print('Data Loading: done')
    return Xtest, Ytest, Ypred


class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self, filename):
        obj = open(filename, 'w')
        pkl.dump(self, obj)


def evaluate_datapoint(root, datapoint):
    if root.data == 'T': return 1
    if root.data == 'F': return 0
    # if datapoint[int(root.data)-1]-1 < 5:
    return evaluate_datapoint(root.nodes[datapoint[int(root.data) - 1] - 1], datapoint)
    # return None


Xtest, Ytest, Ypred = load_data('test.csv', 'output.csv')

Ytree = []
root = pkl.load(open('tree.pkl', 'r'))
for i in range(0, len(Xtest)):
    Ytree.append(evaluate_datapoint(root, Xtest[i]))

accuracy = (len([i for i in range(0, len(Xtest)) if Ytest[i] == Ytree[i]]) + 0.0) / len(Ytest)

print 'Tree prediction accuracy: ', accuracy
accuracy2 = (len([i for i in range(0, len(Xtest)) if Ytest[i] == Ypred[i]]) + 0.0) / len(Ytest)
print 'Output file prediction accuracy: ', accuracy2
ismatch = True
for i in range(0, len(Ytree)):
    if Ytree[i] != Ypred[i]:
        ismatch = False
        break
if (ismatch):
    print('Tree prediction matches output file')
else:
    print('Something is wrong, decision tree classification and output results are not the same')
