import time

import numpy
import generator
if __name__ == '__main__':
    gen = generator.ImageGenerator("C:/Users/Lenovo/PycharmProjects/pythonProject/exercise_data/","C:/Users/Lenovo/PycharmProjects/pythonProject/Labels.json",10,10)
    #time.sleep(3)
    gen.next
    #time.sleep(3)
    #gen.show()
    #gen.class_name(5)
    gen.show()