import os.path
import json
import time

import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        print('***** __init__ *****')
        #TODO: implement constructor
        global filePath
        filePath = file_path
        print("***** File Path is   :   " ,filePath,"\n")

        global labelPath
        labelPath = label_path
        print("***** Label Path is   :   " ,labelPath,"\n")

        global batchSize
        batchSize = batch_size
        print("***** Batch Size is   :   " ,batchSize,"\n")

        global Rotation
        Rotation = rotation
        print("***** Rotation is   :   " ,Rotation,"\n")

        global Mirroring
        Mirroring = mirroring
        print("***** Mirroring is   :   " ,Mirroring,"\n")

        global Shuffle
        Shuffle = shuffle
        print("***** Shuffle is   :   " ,Shuffle,"\n")



        global images2
        images2 = []

        global labels2
        labels2 = []


        global temp
        temp = []

        global images
        images = []

        global labels
        labels = []

        global start
        start = 0






        for size in range(10):
            print("+++Find The Image START+++\n")
            imageName = str(filePath) + str(size) +".npy"

            img = np.load(imageName)

            print("+++RESIZE The Image to 32 * 32 PIXEL+++\n")
            resize(img,(32, 32))# resize the image

            print("+++START PRINT IMAGE++++\n")
            with open('Labels.json', 'r') as f:
                a = json.load(f)
                print(a)
                batchString = str(size)
                print(batchString)
                Label = a[batchString]
                print(Label)
                temp.append([img,Label])

        for item in temp:
            plt.imshow(item[0])
            plt.show()
            print(item[1])


        if Shuffle == True:#
            print("Shuffle START\n\n")
            np.random.shuffle(temp)

            for item in temp:
                plt.imshow(item[0])
                plt.show()
                print(item[1])

            print("Shuffle FINISH\n\n")





    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        #TODO: implement next method
        print("The next() START")
        size = int(batchSize)


        print("+++Batch size is" + str(size) + "\n")


        for item in temp:
            plt.imshow(item[0])
            images.append(item[0])
            plt.show()
            print(item[1])
            labels.append(item[1])



        for batch in range(0,size-1):
            print("***The " + str(batch) +"***\n")



            startpoint = int((start + batch)%99)

            print("++++++++start point+++++" + str(startpoint) + "\n\n")

            images2.append(images[startpoint])



            print("***The " + str(batch) +"FINISH***\n")

            with open('Labels.json', 'r') as f:
                a = json.load(f)
                print(a)
                batchString = str(batch)
                print(batchString)
                Label = a[batchString]
                #print(Label)
                labels2.append(labels[startpoint])

        print("***The next() FINISH")

        return images2 , labels2

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if Mirroring == True:
            print("***Mirroring START***\n")
            plt.imshow(img)
            print("***The original***\n")
            plt.show()
            time.sleep(5)
            imgMirror = img[:, ::-1, :]
            plt.imshow(imgMirror)
            print("***The mirroring***\n")
            plt.show()

            return imgMirror

        if Rotation ==True:
            print("***Rotation START***\n")
            b = np.random.randint(1, 3)
            print("The random number is "+ str(b)+"\n")

            if b == 1:
                print("*** 90 ***\n")
                arr2 = img.copy()
                arr2 = arr2.transpose(1, 0, 2)[::-1]
                arr2 = arr2.reshape(int(arr2.size / 3), 3)
                arr2 = np.array(arr2[::-1])

                arr2 = arr2.reshape(img.shape[1], img.shape[0], img.shape[2])
                plt.imshow(arr2)
                print("*** 90 ***\n")
                plt.show()

            elif b == 2:
                print("*** 180 ***\n")
                arr2 = img.copy()
                arr2 = img.reshape(int(img.size / 3), 3)
                arr2 = np.array(arr2[::-1])
                arr2 = arr2.reshape(img.shape[0], img.shape[1], img.shape[2])
                plt.imshow(arr2)
                print("*** 180 ***\n")
                plt.show()
            else:
                print("*** 270 ***\n")
                arr2 = img.copy()
                arr2 = img.transpose(1, 0, 2)
                plt.imshow(arr2[::-1])
                plt.show()
                print("*** 270 ***\n")

            return arr2




    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        print("+++YOUR INPUT IS "+str(x) + "+++\n")

        with open('Labels.json', 'r') as f:
            a = json.load(f)
            print(a)
            batchString = str(x)
            print(batchString)
            Label = a[batchString]
            print("+++THIS LABEL IS " + str(Label) +"+++\n")





    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method

        print("\033[0;31;40mSHOW TIME\033[0m")
        print("\033[0;32;40mSHOW TIME\033[0m")
        print("\033[0;33;40mSHOW TIME\033[0m")
        print("\033[0;34;40mSHOW TIME\033[0m")
        print("\033[0;35;40mSHOW TIME\033[0m")
        print("\033[0;36;40mSHOW TIME\033[0m")
        print("\033[0;37;40mSHOW TIME\033[0m")

        self.next()

        #for i in range(0,batchSize-1):

        for item in range(0,batchSize-1):
            img = images2[item].copy()
            plt.imshow(img)



            classname = self.class_dict[labels2[item]]
            print("Class Name is " + classname + "\n")






        #for item in temp:
        #    print("The Image is :")
        #    plt.imshow(item[0])
        #    plt.show()

        #    print(item[1])
        #    classname = self.class_dict[item[1]]
        #    print("Class Name is " + classname + "\n")

        #    n=n+1
        #    if n == batchSize:
        #        break

        #print("SHOW FINISH\n")



            #img = temp[0].copy()
            #print("The Image is :")
            #plt.imshow(img)
            #plt.show()

            #classname = self.class_dict[labels[i]]
            #print("Class Name is "+classname + "\n")


