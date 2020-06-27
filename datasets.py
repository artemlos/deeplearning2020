
# this is a template
class Datasets(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class Dataset(Datasets):

    def __init__(self, path):
        self.dataset = read_file(path).split() # also removes whitespace

    # this may not be necessary, since text data is different than image data for instance
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


# class Wikitext2(Datasets):
#
#     def __init__(self, path):
#         self.dataset = read_file(path).split()
#
#     def __getitem__(self, index):
#         return self.dataset[index]
#
#     def __len__(self):
#         return len(self.dataset)
#
# class Enwik9(Datasets):
#
#     def __init__(self, path):
#         self.dataset = read_file(path).split()
#
#     def __getitem__(self, index):
#         return self.dataset[index]
#
#     def __len__(self):
#         return len(self.dataset)

def read_file(path):
    with open(path, encoding='utf8') as fo:
        text = fo.read()
    return text
