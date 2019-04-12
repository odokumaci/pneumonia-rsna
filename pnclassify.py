import os
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import cv2

import matplotlib.pyplot as plt
from keras.models import load_model

import random
import pydicom

# Set seed for reproducibity
np.random.seed(2018)

DATA_DIR = '../input/data/'
train_dicom_dir = os.path.join(DATA_DIR, 'stage_2_train_images')
test_dicom_dir = os.path.join(DATA_DIR, 'stage_2_test_images')

def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations

anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_2_train_labels.csv'))
anns.head()

image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)

ds = pydicom.read_file(image_fps[0])
image = ds.pixel_array # get image array
if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

image_fps_list = list(image_fps)

val_size = round(len(image_fps_list)/10)
test_size = round(len(image_fps_list)/10)
train_size = len(image_fps_list)-test_size-val_size
image_fps_train = image_fps_list[:train_size]
image_fps_val = image_fps_list[train_size:(train_size+val_size)]
image_fps_test = image_fps_list[(train_size+val_size):]

print(len(image_fps_train), len(image_fps_val), len(image_fps_test))

target_val = []
target_train = []
target_test = []
for i in range(len(image_fps_val)):
    target_val.append(image_annotations[image_fps_val[i]][0].Target)

for i in range(len(image_fps_train)):
    target_train.append(image_annotations[image_fps_train[i]][0].Target)

for i in range(len(image_fps_test)):
    target_test.append(image_annotations[image_fps_test[i]][0].Target)

df_valid = pd.DataFrame({'file':image_fps_val,'target':target_val})
df_train = pd.DataFrame({'file':image_fps_train,'target':target_train})
df_test = pd.DataFrame({'file':image_fps_test,'target':target_test})

# Model
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

# create the base pre-trained model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128,128,3))

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)

# final model
model = Model(inputs=base_model.input, outputs=predictions)

#train the model
batch_size = 64

train_datagen = ImageDataGenerator(
#    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


# generators for memory efficient batch processing
def train_generator():
        while True:
            for start in range(0, len(df_train), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_train))
                df_train_batch = df_train[start:end]
                for file, target in df_train_batch.values:
                    ds = pydicom.read_file(file)
                    img = ds.pixel_array
                    img = cv2.resize(img,(128,128))
                    img = np.stack((img,) * 3, -1)
                    img = train_datagen.random_transform(img)
                    x_batch.append(img)
                    y_batch.append(target)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch

def valid_generator():
        while True:
            for start in range(0, len(df_valid), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_valid))
                df_valid_batch = df_valid[start:end]
                for file, target in df_valid_batch.values:
                    ds = pydicom.read_file(file)
                    img = ds.pixel_array
                    img = cv2.resize(img,(128,128))
                    img = np.stack((img,) * 3, -1)
                    x_batch.append(img)
                    y_batch.append(target)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch

def pred_generator():
        while True:
            for start in range(0, len(df_valid), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_valid))
                df_valid_batch = df_valid[start:end]
                for file, target in df_valid_batch.values:
                    ds = pydicom.read_file(file)
                    img = ds.pixel_array
                    img = cv2.resize(img,(128,128))
                    img = np.stack((img,) * 3, -1)
                    x_batch.append(img)
                    y_batch.append(target)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch

def test_pred_generator():
        while True:
            for start in range(0, len(df_test), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_test))
                df_test_batch = df_test[start:end]
                for file, target in df_test_batch.values:
                    ds = pydicom.read_file(file)
                    img = ds.pixel_array
                    img = cv2.resize(img,(128,128))
                    img = np.stack((img,) * 3, -1)
                    x_batch.append(img)
                    y_batch.append(target)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch


# set learning rate schedule
from keras.callbacks import LearningRateScheduler

def lrate_epoch(epoch):
   #epochs_arr = [0, 10, 20, 30]
   epochs_arr = [0, 30, 35, 40]
   learn_rates = [1e-5, 1e-6, 1e-7]
   #epochs_arr = [0, 5]
   #learn_rates = [0.0001]
   lrate = learn_rates[0]
   if (epoch > epochs_arr[len(epochs_arr)-1]):
           lrate = learn_rates[len(epochs_arr)-2]
   for i in range(len(epochs_arr)-1):
       if (epoch > epochs_arr[i] and epoch <= epochs_arr[i+1]):
           lrate = learn_rates[i]
   return lrate

lrateschedule = LearningRateScheduler(lrate_epoch)

# set optimizer and compile model
from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# set callbacks
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='pn_best_da.h5', verbose=1, save_best_only=True)

from keras.callbacks import CSVLogger
csv_logger = CSVLogger('pn_da.csv', append = True)

callbacks = [checkpointer, csv_logger, lrateschedule]


#Finally fit the model
model.fit_generator(generator=train_generator(),
                        steps_per_epoch=(len(df_train) // batch_size) + 1,
                        epochs=40,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator(),
                        validation_steps=(len(df_valid) // batch_size) + 1)

model.save("pn_final_da.h5")

#Reload the model weights and evaluate it on validation set
model.load_weights("pn_best_da.h5")

score = model.evaluate_generator(generator=valid_generator(), steps=(len(df_valid) // batch_size) + 1 )

from sklearn.metrics import fbeta_score

#Validation set auc, roc_plot

p_valid = model.predict_generator(generator=pred_generator(), steps=(len(df_valid) // batch_size) + 1 )

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(target_val, p_valid)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(target_val, p_valid)
plt.title('ROC Curve')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.axis([0,1,0,1])
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()

#Test set auc, roc_plot

p_test = model.predict_generator(generator=test_pred_generator(), steps=(len(df_test) // batch_size) + 1 )

auc_test = roc_auc_score(target_test, p_test)
print('AUC: %.3f' % auc_test)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(target_test, p_test)
plt.title('ROC Curve')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.axis([0,1,0,1])
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')

# show the plot
plt.show()
