import codecs
import json
import pickle

class Dict(object):
    def __init__(self, lower=False):
        print("init Dict")

        self.labelToIdx = {"!": 0, "#":1, "$":2, "%":3, "&":4, "'":5, "<NULL>":6, "added":7, "_":8, "removed":9}
        self.idxToLabel = {0:"!", 1: "#", 2:"$", 3:"%", 4:"&", 5:"'", 6:"<NULL>", 7:"added", 8:"_", 9:"removed"}

        self.frequencies = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = [0,1,2,3,4,5,6,7,8,9]

    def size(self):
        return len(self.idxToLabel)

    def writeFileDict(self, filename):
        pickle.dump(self.labelToIdx, open(filename, 'wb'))

    def get_dict(self):
        return self.labelToIdx

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    def addSpecial(self, label, idx=None):
        "Mark this `label` and `idx` as special (i.e. will not be pruned)."
        idx = self.add(label, idx)
        self.special += [idx]

    def addSpecials(self, labels):
        "Mark all labels in `labels` as specials (i.e. will not be pruned)."
        for label in labels:
            self.addSpecial(label)

    def add(self, label, idx=None):
        "Add `label` in the dictionary. Use `idx` as its index if given."
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    def prune(self, size):
        "Return a new dictionary with the `size` most frequent entries."
        if size >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = [(i,self.frequencies[i]) for i in range(10,len(self.frequencies))]
        freq.sort(key=lambda x:x[1], reverse=True)
        newDict = Dict()
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i, _ in freq[:size]:
            newDict.add(self.idxToLabel[i])

        return newDict

