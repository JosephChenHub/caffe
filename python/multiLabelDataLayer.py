
import caffe
import numpy as np
import random
from PIL import Image
import scipy.misc
import pdb

'''
 this layer is created for reading such data format:
     img1, 0, 1, 1, 2, ..., 0
     img2, 2, 3, 0, 2, ..., 2
     ...
 multi-label can be real value, i.e. data type is float32
'''
'''
    Setup data layer according to parameters:
        - split: train/val/test
        - batch_size
        - mean: tuple of mean values to subtract
        - shuffle: load in random order(default: True)
        - seed: seed for randomization (default: None or current time)
    Usage example:
        params = dict( split = 'multi_label_train.txt',
                 batch_size = 128,
                 im_shape = (256, 256)
                 transformer = dict(
                    mean = (104.002, 118.233, 120.002),
                    )
                 shuffle = True
                 )
'''
class MultiLabelDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)

        self.split = params['split']
        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']
        self.transformer = params.get('transformer', None)
        self.shuffle = params.get('shuffle', True)
        self.c = params.get('num_output', 102)
        #check num of blobs
        if len(top) != 2:
            raise Exception('FATAL:Need to define two top blobs: data and label!')
        if len(bottom) != 0:
            raise Exception('FATAL:Do not define a bottom blob')

        self.indices = [line.rstrip() for line in open(self.split, 'r')]

        if self.batch_size > len(self.indices):
            raise Exception('FATAL: Batch size is greater than num of samples!')
        self.idx = 0
        print('Total set:{} images\n'.format(len(self.indices)))

        if 'train' not in self.split:
            self.shuffle = False

        top[0].reshape(self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        top[1].reshape(self.batch_size, self.c,1 ,1)
        self.cnt = 0

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        #output blobs
        for itt in range(self.batch_size):
            self.cnt += 1
            if self.cnt == 288:
                pdb.set_trace()
            data, label = self.load_next_img()
            top[0].data[itt, ...] = data
            top[1].data[itt, ...] = label.reshape(self.c, 1, 1)

    def backward(self, top, propagate_down, bottom):
        pass

    def load_next_img(self):
        #Did we finish an epoch ?
        if self.idx == len(self.indices):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.indices)
        # load an image
        index = self.indices[self.idx]
        split_str = index.split(',')
        img_file = split_str[0]
        img_src = Image.open(img_file)

        img = np.asarray(img_src, dtype = np.float32)
        img = img[:, :, ::-1] #RGB->BGR
        if img.shape[:2] != self.im_shape:
            img = scipy.misc.imresize(img, self.im_shape) #resize
        # transformer
        if self.transformer:
            mean = self.transformer.get('mean', None)
            if mean:
                img -= mean
        img = img.transpose((2,0,1)) #HxWxC->CxHxW
        # multi-label
        c = len(split_str)-1
        label = np.zeros(c, dtype = np.float32)
        for i in range(1, c):
            label[i] = np.float32(split_str[i])

        self.idx += 1

        return img, label

