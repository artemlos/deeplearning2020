import os
import tensorflow as tf
import requests
import zipfile
import io

# by default every dataset class downloads the data

class PTB:

    def __init__(self, task=None, local=False):
        # local dataset file paths
        ptb_char_train_path = os.path.join("datasets", "ptb", "ptb.char.train.txt")
        ptb_char_valid_path = os.path.join("datasets", "ptb", "ptb.char.valid.txt")
        ptb_word_train_path = os.path.join("datasets", "ptb", "ptb.train.txt")
        ptb_word_valid_path = os.path.join("datasets", "ptb", "ptb.valid.txt")
        ptb_word_test_path = os.path.join("datasets", "ptb", "ptb.test.txt")

        if not local:
            ptb_char_train_path = tf.keras.utils.get_file("ptb.char.train.txt", "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.char.train.txt")
            ptb_char_valid_path = tf.keras.utils.get_file("ptb.char.valid.txt", "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.char.valid.txt")
            ptb_word_train_path = tf.keras.utils.get_file("ptb.train.txt", "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt")
            ptb_word_valid_path = tf.keras.utils.get_file("ptb.valid.txt", "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt")
            ptb_word_test_path = tf.keras.utils.get_file("ptb.test.txt", "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt")

        if not task:
            self.char_train = read_file(ptb_char_train_path)
            self.char_valid = read_file(ptb_char_valid_path)
            self.word_train = read_file(ptb_word_train_path)
            self.word_valid = read_file(ptb_word_valid_path)
            self.word_test = read_file(ptb_word_test_path)
        elif task == "char":
            self.char_train = read_file(ptb_char_train_path)
            self.char_valid = read_file(ptb_char_valid_path)
        elif task == "word":
            self.word_train = read_file(ptb_word_train_path)
            self.word_valid = read_file(ptb_word_valid_path)
            self.word_test = read_file(ptb_word_test_path)



class Wikitext2:

    def __init__(self, local=False):
        # local file paths
        wikitext2_train_path = os.path.join("datasets", "wikitext-2-v1", "wikitext-2", "wiki.train.tokens")
        wikitext2_valid_path = os.path.join("datasets", "wikitext-2-v1", "wikitext-2", "wiki.valid.tokens")
        wikitext2_test_path = os.path.join("datasets", "wikitext-2-v1", "wikitext-2", "wiki.test.tokens")

        if not local:
            wikitext2_unzipped = tf.keras.utils.get_file("wikitext-2-v1", "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
                                                         extract=True, archive_format='zip')
            wikitext2_unzipped = wikitext2_unzipped.replace("\\wikitext-2-v1", '') # this is because the extracted folder name is different than zipped folder
            wikitext2_train_path = os.path.join(wikitext2_unzipped, "wikitext-2", "wiki.train.tokens")
            wikitext2_valid_path = os.path.join(wikitext2_unzipped, "wikitext-2", "wiki.valid.tokens")
            wikitext2_test_path = os.path.join(wikitext2_unzipped, "wikitext-2", "wiki.test.tokens")

        self.train = read_file(wikitext2_train_path)
        self.valid = read_file(wikitext2_valid_path)
        self.test = read_file(wikitext2_test_path)


class Enwik9:

    def __init__(self, local=False):
        # local file path
        enwik9_path = os.path.join("datasets", "enwik9", "enwik9")

        if not local:
            # change file destination if desire to download it somewhere else
            file_dest = os.path.join("C:\\", "Users", "harry", ".keras", "datasets")
            enwik9_path = os.path.join(file_dest, "enwik9")
            if not os.path.isfile(enwik9_path):
                zipfile.ZipFile(
                    io.BytesIO(requests.get("http://mattmahoney.net/dc/enwik9.zip", stream=True).content)).extractall(file_dest)

        self.enwik9_train, self.enwik9_valid, self.enwik9_test = self.__split_training_data(read_file(enwik9_path))

    def __split_training_data(self, enwik9):
        train_offset = 9*10**7
        valid_offset = train_offset + 5*10**6
        test_offset = valid_offset + 5*10**6
        enwik9_train = enwik9[:train_offset] # first 90 million for training
        enwik9_valid = enwik9[train_offset: valid_offset] # 5 million for valid
        enwik9_test = enwik9[valid_offset: test_offset] # 5 million for test

        return enwik9_train, enwik9_valid, enwik9_test

def read_file(path):
    with open(path, encoding='utf8') as fo:
        text = fo.read()
    return text.split()  # remove whitespaces
