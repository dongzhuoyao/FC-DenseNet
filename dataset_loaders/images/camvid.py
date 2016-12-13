import cv2
import numpy as np
from  numpy.random import RandomState
from random import shuffle
import os

class CamvidDataset():
    def __init__(self,
                 which_set,
                 batch_size,
                 seq_per_video,
                 seq_length,
                 crop_size,
                 get_one_hot,
                 get_01c,
                 overlap,
                 use_threads,
                 shuffle_at_each_epoch = False,
                 horizontal_flip=False,
                 save_to_dir=False):


        self.which_set = which_set
        self.batch_size = batch_size

        self.crop_size = crop_size
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]

        self.overlap = overlap
        self.shuffle_at_each_epoch = shuffle_at_each_epoch
        self.rng =RandomState(0)

        self.img_height = 360
        self.img_width = 480

        self.init_lines()
        self.line_index = 0
        self.data_shape = (3,224,224)


    def init_lines(self):
        data_line = []

        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        data_file = open(os.path.join(__location__, "CamVid/"+self.which_set+".txt"),"r")
        for line in data_file.readlines():
            line = line.strip("\n")
            line_arr = line.split(" ")
            data_line.append(line_arr)
            data_file.close()

        self.data_line = data_line
        self.data_line_num = len(data_line)
        shuffle(self.data_line)



    def get_n_classes(self):
        return 12

    def get_n_samples(self):
        if self.which_set=="train":
            return  367
        if self.which_set=="test":
            return 233
        if self.which_set=="val":
            return 101

    def get_n_batches(self):

        if self.which_set=="train":
            return  367/self.batch_size
        if self.which_set=="test":
            return 233/self.batch_size
        if self.which_set=="val":
            return 101/self.batch_size

    def get_void_labels(self):
        return [0]

    def next(self):

        X = np.zeros((self.batch_size, 3, self.crop_height, self.crop_width), dtype="float32")
        y = np.zeros((self.batch_size,self.crop_height, self.crop_width), dtype="int32")
        for i in xrange(self.batch_size):
            cv_img = cv2.imread(self.data_line[i][0])
            cv_lab = cv2.imread(self.data_line[i][1])

            h_off = np.random.random_integers(1000) % (self.img_height - self.crop_height + 1)
            w_off = np.random.random_integers(1000) % (self.img_width - self.img_width + 1)

            cv_img = cv_img[h_off:h_off+self.crop_height, w_off:w_off+self.crop_width]
            cv_lab = cv_lab[h_off:h_off+self.crop_height, w_off:w_off+self.crop_width]

            #random flip
            if self.rng.randint(1,10)%2:
                cv2.flip(cv_img, cv_img, 1)
                cv2.flip(cv_lab, cv_lab, 1)


            X[i,:,:,:] = cv_img
            y[i,:,:] = cv_lab

            self.line_index += 1
            if self.line_index >=self.data_line_num:
                print "Restarting data prefetching from start"
                self.line_index = 0
                shuffle(self.data_line)
        return X,y

