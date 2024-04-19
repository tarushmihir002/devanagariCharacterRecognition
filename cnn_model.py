import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"



dataset = pd.read_csv("devanagiri_data.csv")


x = dataset.values[:,:-1] / 255.0
y = dataset['character'].values  

del dataset

no_of_classes = 46 

img_width= 32
img_height= 32


"""uncomment the below code snippet to visulaize some datapoints of out dataset"""

# cutsomcmap = sns.dark_palette("white", as_cmap=True)
# random_idxs = random.sample(range(1, len(y)), 4)
# plt_dims = (15, 2.5)
# f, axarr = plt.subplots(1, 4, figsize=plt_dims)
# it = 0
# for idx in random_idxs:
#     image = x[idx, :].reshape((img_width_cols, img_height_rows)) * 255
#     axarr[it].set_title(y[idx])
#     axarr[it].axis('off')
#     sns.heatmap(data=image.astype(np.uint8), cmap=cutsomcmap, ax=axarr[it])
#     it = it+1
# plt.show()




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = to_categorical(y_train, no_of_classes)
y_test = to_categorical(y_test, no_of_classes)


im_shape = (img_height, img_width, 1)
x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)


"Define the CNN model"
cnn = Sequential()

kernelSize = (3, 3)
ip_activation = 'relu'
ip_conv_0 = Conv2D(filters=32, kernel_size=kernelSize, input_shape=im_shape, activation=ip_activation)
cnn.add(ip_conv_0)


ip_conv_0_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_0_1)


pool_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_0)


ip_conv_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1)
ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1_1)

pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_1)



drop_layer_0 = Dropout(0.2)
cnn.add(drop_layer_0)

flat_layer_0 = Flatten()
cnn.add(Flatten())



h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_0)

h_dense_1 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_1)



op_activation = 'softmax'
output_layer = Dense(units=no_of_classes, activation=op_activation, kernel_initializer='uniform')
cnn.add(output_layer)

# print(cnn.summary())

opt = 'adam'
# opt='SGD'
# opt='RMSprop'

loss = 'categorical_crossentropy'
metrics = ['accuracy']

cnn.compile(optimizer=opt, loss=loss, metrics=metrics)

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)


history = cnn.fit(x_train,y_train,
                  batch_size=32, epochs=3,
                  validation_data=(x_test, y_test))

scores = cnn.evaluate(x_test, y_test, verbose=0)
print("Hiustotoe")
print(history)

print("hisot.hsit")
print(history.history)
print(f"Accuracy:{scores[1]*100}")


figure_1, axis_accuracy = plt.subplots()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy of the CNN Model')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()



figure_2, axis_loss = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss of the CNN Model')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
