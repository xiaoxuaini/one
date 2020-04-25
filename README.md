# one
lenarn and practice


#自己练习，4.25整理
import tensorflow as tf

    

mnist = tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels) = mnist.load_data()

training_images = training_images.reshape(60000,28,28,1)
training_images = training_images/255.0 
test_images = test_images.reshape(10000,28,28,1)
test_images = test_images /255.0

#卷积模型model，
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3), activation='relu',#第一个卷积，生成64个3*3的过滤器，
                           input_shape=(28,28,1)),#输入形状为28*28
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128,activation='relu'),#隐藏层有128个神经单元
    tf.keras.layers.Dense(units=10,activation='softmax')#输出层有10个单元
    ])

model.compile(optimizer = tf.optimizers.Adam(),#结合优化器和损失函数
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])




#观察卷积和池化
print(test_labels[:100])#打印出测试集的前100个元素
import matplotlib.pyplot as plt#显示图表的一种函数
f,axarr = plt.subplots(3,4)#控制图表的排列，几行几列
FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=28
CONVOLUTION_NUMBER = 5#？？？？
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model =tf.keras.models.Model(inputs =model.input,outputs = layer_outputs)
for x in range(0,4):#循环语句
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f1[0,:,:,CONVOLUTION_NUMBER],cmap='inferno')
    axarr[0,x].grid(False)
    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f2[0,:,:,CONVOLUTION_NUMBER],cmap='inferno')
    axarr[0,x].grid(False)
    f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f3[0,:,:,CONVOLUTION_NUMBER],cmap='inferno')
    axarr[0,x].grid(False)
