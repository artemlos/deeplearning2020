import os
import tensorflow as tf
import requests
import zipfile
import io
import numpy as np

# by default every dataset class downloads the data
fname_to_url = {
    "ptb.char.train": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.char.train.txt",
    "ptb.char.valid": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.char.valid.txt",
    "ptb.train": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt",
    "ptb.valid": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt",
    "ptb.test": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt",
    "wikitext-2-v1": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
    "enwik9": "http://mattmahoney.net/dc/enwik9.zip"
}


class Dataset:
    def __init__(self, fname, f_dest=""):
        if "ptb" in fname:
            self.data = self.__read_file(tf.keras.utils.get_file(fname + ".txt", fname_to_url[fname]))
        elif "wiki" in fname:
            wikitext2_unzipped = tf.keras.utils.get_file("wikitext-2-v1", fname_to_url["wikitext-2-v1"], extract=True, archive_format='zip')
            wikitext2_unzipped = wikitext2_unzipped.replace("\\wikitext-2-v1", '')  # this is because the extracted folder name is different than zipped folder
            self.data = self.__read_file(os.path.join(wikitext2_unzipped, "wikitext-2", fname + ".tokens"))
        elif "enwik" in fname:
            # change file destination if desire to download it somewhere else
            file_dest = os.path.join(f_dest)
            enwik9_path = os.path.join(file_dest, "enwik9")
            # check if file not already exists
            if not os.path.isfile(enwik9_path):
                zipfile.ZipFile(
                    io.BytesIO(requests.get(fname_to_url["enwik9"], stream=True).content)).extractall(file_dest)

            train_offset = 9*10**7
            valid_offset = train_offset + 5*10**6
            test_offset = valid_offset + 5*10**6
            enwik = self.__read_file(enwik9_path)
            if "train" in fname:
                self.data = enwik[:train_offset]  # first 90 million for training
            elif "valid" in fname:
                self.data = enwik[train_offset: valid_offset]  # 5 million for valid
            elif "test" in fname:
                self.data = enwik[valid_offset: test_offset]  # 5 million for test

        self.char2idx = {u:i for i,u in enumerate(sorted(set(self.data)))}
        self.idx2char = np.array(len(self.char2idx))

    def convert_text_to_int(self):
        self.data = np.array([self.char2idx[c] for c in self.data])
        return self.data

    def convert_to_tensor_dataset(self):
        self.data = tf.data.Dataset.from_tensor_slices(self.data)
        return self.data

    def batch(self, batch_size, drop_remainder):
        self.data = self.data.batch(batch_size, drop_remainder=drop_remainder)
        return self.data

    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
        self.data = self.data.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        return self.data

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    def __read_file(self, path):
        with open(path, encoding='utf8') as fo:
            text = fo.read()
        return text.split()  # remove whitespaces
