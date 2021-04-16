import skimage
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    img = np.load("C:/Users/Lenovo/PycharmProjects/pythonProject/exercise_data/91.npy")
    plt.imshow(img)
    plt.show()
    imgMirror = img[:,::-1,:]
    plt.imshow(imgMirror)
    plt.show()

    #逆时针旋转90度
    img90 = img.copy()
    img90 = img90.transpose(1, 0, 2)[::-1]
    img90 = img90.reshape(int(img90.size / 3), 3)
    img90 = np.array(img90[::-1])
    # 恢复原数组维度，这个需要注意，图像长宽尺寸与原图相反；
    arr2 = img90.reshape(img.shape[1], img.shape[0], img.shape[2])
    plt.imshow(arr2)
    plt.show()
    print("90\n")

    img270 = img.copy()
    img270 = img270.transpose(1, 0, 2)  # 行列转置
    plt.imshow(img270[::-1])
    plt.show()
    print("270\n")

    img180 = img[::-1]
    plt.imshow(img180)
    plt.show()

    img90 = img