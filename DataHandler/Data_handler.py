from scipy.ndimage import rotate
import os, glob
import numpy as np
import cv2

__author__ = 'tylin'
__version__ = 0.9
import json
import datetime
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import cv2
import os


class Datahandler_COCO():
    def __init__(self,image_dir,annotation_file):
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()


        self.annotation_file=annotation_file
        self.image_dir = image_dir
        print("loading dataset")

        dataset = json.load(open(self.annotation_file, 'r'))

        self.coco = COCO(self.annotation_file)
        # Load all classes (Only Building in this version)
        self.classIds = self.coco.getCatIds()
        #print(self.classIds)
        # Load all images
        self.image_ids = list(self.coco.imgs.keys())
        #print(self.image_ids)
        for image_id in self.image_ids:
            self.anns[image_id]=[]

        self.categories=self.coco.loadCats([100])
        #print(self.categories)
        for object in dataset["annotations"]:
            self.anns[object["image_id"]].append(object)
            #print (object)
        #print(self.anns)
    def get_mask(self,id):

        temp=self.anns[id]
        m=self.coco.annToMask(temp[0])
        for ob in temp[1:]:
            m1=self.coco.annToMask(ob)
            m=m|m1
        #print(temp)
        return m

    def make_batches(self,batchsize=4,Train=True):
        batch_images = []
        batch_masks = []
        list = self.image_ids
        #print(self.image_ids)

        while True:
            for id in list:
                #print(id)
                filename=self.coco.imgs[id]["file_name"]
                path=os.path.join(self.image_dir,filename)
                print(path)
                img=cv2.imread(path)
                mask=self.get_mask(id)
                img=cv2.resize(img,(224,224))
                mask=cv2.resize(mask,(224,224))

                batch_images.append(img)
                batch_masks.append(mask)

                #cv2.imwrite(filename+"gt.jpg",img)
                #cv2.imwrite(filename+"b.jpg",mask*255)
                if len(batch_images) == batchsize:

                    yield (np.array(batch_images), np.expand_dims(np.array(batch_masks),axis=-1))
                    batch_images = []
                    batch_masks = []

    def get_batch(self,batch_size=1, train=True):
        a=next(self.make_batches(batch_size,train))
        for b in a:
            print(b)
            return np.array(b),np.expand_dims(np.array(b),axis=-1)

        #self.coco.
        # register classes


class Data_handler:
    def __init__(self,
                 data_location,
                 ground_truth):
        # os.chdir(data_location)
        self.images = []
        self.GTs = []
        self.data_path = data_location
        self.labels_path = ground_truth

        for root, dirs, files in os.walk(data_location):
            for file in files:
                if file.endswith((".png", ".tif")):  # The arg can be a tuple of suffixes to look for

                    self.images.append(file)
                    gt_name = file.split(".")[0][:] + "_L.png"
                    self.GTs.append(gt_name)

                if file.endswith((".JPG")):  # The arg can be a tuple of suffixes to look for
                    self.images.append(file)
                    gt_name = file.split(".")[0][:-2] + "_L.png"
                    self.GTs.append(gt_name)

        self.data_length = len(self.images)
        self.current_location = 0
        return

    def get_batch(self, batch_size, train=True):
        final_images = np.zeros((batch_size, 416, 416, 6))
        final_GT = np.zeros((batch_size, 416, 416))
        for i in range(batch_size):
            image_path = self.data_path + self.images[self.current_location]
            image = cv2.imread(image_path)

            gt_path = self.labels_path + self.GTs[self.current_location]
            print(image_path)
            print(gt_path)
            GT = cv2.imread(gt_path)

            image_six_chl = np.zeros((416, 416, 6))
            GT_one_chl = np.zeros((GT.shape[0], GT.shape[1]))
            GT_one_chl[GT[:, :, 1] < 27] = 1
            GT_one_chl[(GT[:, :, 1] > 127) * (GT[:, :, 2] > 127)] = 2

            if train:
                CLAHE_contrast_range = (1, 11)
                image_crop_range = (500, 800)
                CLAHE_contrast__window_range = (8, 40)
                rot_angle = [0, 90, 180, 270]

                angle = rot_angle[np.random.randint(0, 4)]

                resize_number = np.random.randint(image_crop_range[0], image_crop_range[1])
                if (np.random.randint(0, 100) % 3 == 0):
                    image = cv2.resize(image, (resize_number, resize_number))
                    GT_one_chl = cv2.resize(GT_one_chl, (resize_number, resize_number))

                    crop_upper_limit = resize_number - 416 - 1
                    crop_index_x = np.random.randint(0, crop_upper_limit)
                    crop_index_y = np.random.randint(0, crop_upper_limit)

                    image = image[crop_index_x:crop_index_x + 416, crop_index_y:crop_index_y + 416]
                    GT_one_chl = GT_one_chl[crop_index_x:crop_index_x + 416, crop_index_y:crop_index_y + 416]
                else:
                    image = cv2.resize(image, (416, 416))
                    GT_one_chl = cv2.resize(GT_one_chl, (416, 416))

                window_size = np.random.randint(CLAHE_contrast__window_range[0], CLAHE_contrast__window_range[1])
                contrast_factor = np.random.randint(CLAHE_contrast_range[0], CLAHE_contrast_range[1])

                clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(window_size, window_size))
                image = rotate(image, angle)
                GT_one_chl = rotate(GT_one_chl, angle)
                b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]

                cl1 = clahe.apply(b)
                cl2 = clahe.apply(g)
                cl3 = clahe.apply(r)
                image_six_chl[:, :, 0:3] = image

                image_six_chl[:, :, 3] = cl1
                image_six_chl[:, :, 4] = cl2
                image_six_chl[:, :, 5] = cl3

            else:
                image = cv2.resize(image, (416, 416))
                GT_one_chl = cv2.resize(GT_one_chl, (416, 416))

                b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]

                clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(20, 20))

                cl1 = clahe.apply(b)
                cl2 = clahe.apply(g)
                cl3 = clahe.apply(r)
                image_six_chl[:, :, 0:3] = image

                image_six_chl[:, :, 3] = cl1
                image_six_chl[:, :, 4] = cl2
                image_six_chl[:, :, 5] = cl3

            final_images[i] = image_six_chl
            final_GT[i] = GT_one_chl
            self.current_location += 1
            if (self.current_location > (self.data_length - 1)):
                self.current_location = 0

        return [final_images, final_GT]
