from posixpath import join
import torch.utils.data as data
import numpy as np
import pickle


class LMDBDataset(data.Dataset):
    def __init__(self, db_path, size=None, repeat=1):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            length = txn.stat()['entries']
        # cache_file = '_cache_' + db_path.replace('/', '_')
        # if os.path.isfile(cache_file):
        #     self.keys = pickle.load(open(cache_file, "rb"))
        # else:
        #     with self.env.begin(write=False) as txn:
        #         self.keys = [key for key, _ in txn.cursor()]
        #     pickle.dump(self.keys, open(cache_file, "wb"))
        self.length = size or length
        self.repeat = repeat
        self.meta = pickle.load(open(join(db_path, 'meta_info.pkl'), 'rb'))
        self.shape = self.meta['shape']
        self.dtype = self.meta['dtype']

    def __getitem__(self, index):
        env = self.env
        index = index % self.length
        
        with env.begin(write=False) as txn:
            raw_data = txn.get('{:08}'.format(index).encode('ascii'))

        flat_x = np.frombuffer(raw_data, self.dtype)
        x = flat_x.reshape(*self.shape)
        
        if self.dtype == np.uint16:
            x = np.clip(x / 65535, 0, 1).astype(np.float32)

        return x

    def __len__(self):
        return int(self.length * self.repeat)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
