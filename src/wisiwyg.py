import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, \
    ZeroPadding2D, MaxPooling2D, AveragePooling2D, \
    Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt
from os import listdir
import socket
import time
import pandas as pd
from pandas import ExcelFile
import numpy as np
import random
import sys, urllib.request
from pandas import ExcelWriter

import threading

# region TCP Connections
TCP_IP = '127.0.0.1'
TCP_PORT = 7001
BUFFER_SIZE = 512

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

conn, addr = s.accept()

# print connection address when someone connects
print('Connection address:', addr)
# endregion

face_cascade = cv2.CascadeClassifier('../haarcascade_files/haarcascade_frontalface_default.xml')
df1 = pd.read_excel('../data/getty_dataset.xlsx', sheet_name='Generated Descriptions')
df2 = pd.read_excel('../data/getty_dataset.xlsx', sheet_name='Generated Titles')
# endregion

#region Setup for Get Title by index
indexNames_title = df2[df2['Reject'].isnull() == False].index
df2.drop(indexNames_title,inplace=True)
df2_title = df2['Title']
df_numOfPeople_title = df2['# of People']

newdf2 = pd.concat([df2_title,df_numOfPeople_title],axis=1)
newdf2 = newdf2.dropna()
df2_title = newdf2['Title']

with open('titles.txt', 'w') as f:
    for item in df2_title.values.tolist():
        f.write("%s\n" % item)

titles_list = open("titles.txt", 'r').read().split('\n')

df_numOfPeople_title = newdf2['# of People']

is_single_title = df_numOfPeople_title.astype(str).str.contains('1')
is_multiple_title = df_numOfPeople_title.astype(str).str.contains('2') | df_numOfPeople_title.astype(str).str.contains('3+')
df_single_title = df_numOfPeople_title.loc[is_single_title].index
df_multiple_title = df_numOfPeople_title.loc[is_multiple_title].index
# endregion

# region Setup for Get Description by Index
indexNames = df1[df1['Reject'].isnull() == False].index
df1.drop(indexNames, inplace=True)

df_ageRange = df1['Age Range']
df_numOfPeople = df1['# of People']
df_emotion = df1['Prompt']
df_responses = df1['Edited Response']

newdf1 = pd.concat([df_ageRange, df_numOfPeople,df_emotion,df_responses],axis=1)
newdf1 = newdf1.dropna()
df_ageRange = newdf1['Age Range']
df_numOfPeople = newdf1['# of People']
df_emotion = newdf1['Prompt']
df_responses = newdf1['Edited Response']

with open('../data/descriptions.txt', 'w') as f:
    for item in df_responses.values.tolist():
        content ="%s\n" % item
        f.write(content.encode("gbk", 'ignore').decode("gbk", "ignore"))

descriptions_list = open("../data/descriptions.txt", 'r').read().split('\n')

is_child = df_ageRange.str.contains("child")
is_young = df_ageRange.str.contains("young")
is_old = df_ageRange.str.contains("old")
is_single = df_numOfPeople.astype(str).str.contains('1')
is_multiple = df_numOfPeople.astype(str).str.contains('2') | df_numOfPeople.astype(str).str.contains('3+')
is_happy = df_emotion.str.contains("happ")
is_sad = df_emotion.str.contains("sad")
is_angry = df_emotion.str.contains("angr")
is_surprise = df_emotion.str.contains("surpri")
is_disgust = df_emotion.str.contains("disgus")
is_fear = df_emotion.str.contains("fear")
is_netural = (is_happy|is_sad|is_angry|is_surprise|is_disgust|is_fear) == False

df_child = df_ageRange.loc[is_child].index
df_young = df_ageRange.loc[is_young].index
df_old = df_ageRange.loc[is_old].index

df_single = df_numOfPeople.loc[is_single].index
df_multiple = df_numOfPeople.loc[is_multiple].index

df_happy = df_emotion.loc[is_happy].index
df_sad = df_emotion.loc[is_sad].index
df_angry = df_emotion.loc[is_angry].index
df_surprise = df_emotion.loc[is_surprise].index
df_disgust = df_emotion.loc[is_disgust].index
df_fear = df_emotion.loc[is_fear].index
df_neutral=  df_emotion.loc[is_netural].index
# endregion

# region Helper Functions
def get_description_by_index(indexs):
    return df_responses[random.choice(indexs)]

def get_title_by_index(indexs):
    return df2_title[random.choice(indexs)]

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def loadEmotionModel():
    # define CNN model
    model = Sequential()

    # 1st convolution layer
    model.add(Convolution2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    num_classes = 7
    model.add(Dense(num_classes, activation='softmax'))
    return model

def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model

def ageModel():
    model = loadVggFaceModel()

    base_model_output = Sequential()
    base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    age_model = Model(inputs=model.input, outputs=base_model_output)
    age_model.load_weights("../models/age_model_weights.h5")
    return age_model

def genderModel():
    model = loadVggFaceModel()

    base_model_output = Sequential()
    base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    gender_model = Model(inputs=model.input, outputs=base_model_output)
    gender_model.load_weights("../models/gender_model_weights.h5")
    return gender_model

def emotionModel():
    # emotion_model = model_from_json(open("models/facial_expression_model_structure.json", "r").read())
    emotion_model = loadEmotionModel()
    emotion_model.load_weights('../models/facial_expression_model_weights.h5')  # load weights
    return emotion_model

# send json to touch designer
def jsonSender():
    #threading.Timer(5.0, jsonSender).start()
    print("send message..")

    json_str = '[{"numPep": "' + str(numPep) + \
               '", "emotion01": "' + emotion01 + \
               '", "emotion02": "' + emotion02 + \
               '", "gender01": "' + gender01 +\
               '", "gender02": "' + gender02 +\
               '", "age01": "' + str(age01) +\
               '", "age02": "' + str(age02) +\
               '", "title": "' + title +\
               '", "description": "' + description +'"}]\r\n'

    print(json_str)
    conn.send(json_str.encode('utf-8'))

# return the most frequency
def mostFrequent(arr, n):
    arr.sort()
    max_count = 1
    res = arr[0]
    curr_count = 1

    for i in range(1, n):
        if (arr[i] == arr[i - 1]):
            curr_count += 1
        else:
            if (curr_count > max_count):
                max_count = curr_count
                res = arr[i - 1]
            curr_count = 1
    if (curr_count > max_count):
        max_count = curr_count
        res = arr[n - 1]
    return res

def getMostFrequentElement(choosenIndex, defaultReturn):
    return defaultReturn if len(choosenIndex) == 0 else mostFrequent(choosenIndex, len(choosenIndex))

def getGender(choosenIndex):
    return "male" if choosenIndex==1 else "female"

def getAgeRange(age):
    if(0<=age<=17):
        return 'child'
    elif(17<age<=25):
        return 'young'
    elif(age>17):
        return 'old'



# endregion

# region Load Trained Model
age_model = ageModel()
gender_model = genderModel()
emotion_model = emotionModel()
# endregion

# region Opencv Detection
# age model has 101 outputs and its outputs will be multiplied by its index label.
# sum will be apparent age
output_indexes = np.array([i for i in range(0, 101)])

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
emotionsIndex = [df_angry,df_disgust,df_fear,df_happy,df_sad,df_surprise,df_neutral]
shouldUpdateEmotion = False

# Prediction Lists for Getting the Most Frequent Emotion age gender
max_emotionIndex1 = []
max_emotionIndex2 = []
max_genderIndex1 = []
max_genderIndex2 = []

#endregion

# region Timer Variables
currentTime = 0.0
peopleLeaveTime=0.0
jsonSendingInterval = 4.0
# endregion

# region Opencv Declaration
cap = cv2.VideoCapture(0)  # capture webcam
# endregion

# region JSON DATA
numPep = 0
emotion01 = "neutral"
emotion02 = "neutral"
gender01 = "female"
gender02 = "female"
age01 = ''
age02 = ''
title = "This is a Title"
description = "This is a description!!!!!!!!!!!!!!!"
# endregion


while True:

    ret, img = cap.read()
    # img = cv2.resize(img, (640, 360))

    '''Video Cropping'''
    img = img[0:640,0:500]
    canvas = np.zeros((640, 480, 3), dtype="uint8")
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    # For some reason when no faces detected, it is a tuple, when there is faces detected, it become a np array
    if (type(faces) is not tuple):
        print("Number of faces: ", len(faces))
        numPep = len(faces)
    else:  # so when it is a tuple which means no people are detected
        numPep = 0


    # Timer for update the emotion
    if currentTime < time.time():
        currentTime = time.time() + jsonSendingInterval  # reset timer
        shouldUpdateEmotion = True

    # Reset and send the number of people that are detected after certain period
    if numPep == 0:
        if time.time() - peopleLeaveTime > jsonSendingInterval:
            peopleLeaveTime= time.time()
            emotion01 = "neutral"
            emotion02 = "neutral"
            age01 = 'young'
            age02 = 'young'
            gender01 = "female"
            gender02 = "female"
            title = random.choice(titles_list)
            description = random.choice(descriptions_list)
            jsonSender()
    else:
        peopleLeaveTime =time.time()# reset timer


    predictions = [] # a list for saving multiple emotions
    for (x, y, w, h) in faces:
        if w > 40:  # 130: #ignore small faces

            # extract detected face
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face

            # detect emotions
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize all pixels to scale of [0, 1]

            emotion_distributions = emotion_model.predict(img_pixels)
            age_distributions = 0
            gender_distribution=0

            try:
                # age gender data set has 40% margin around the face. expand detected face.
                margin = 30
                margin_x = int((w * margin) / 100)
                margin_y = int((h * margin) / 100)
                detected_face = img[int(y - margin_y):int(y + h + margin_y), int(x - margin_x):int(x + w + margin_x)]
            except:
                print("detected face has no margin")
            try:
                detected_face = cv2.resize(detected_face, (224, 224))

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                # find out age and gender
                age_distributions = age_model.predict(img_pixels)
                gender_distribution = gender_model.predict(img_pixels)[0]
            except Exception as e:
                print("exception", str(e))


            data = [] # a list to save all detections in current loop
            data.append(emotion_distributions)
            data.append(age_distributions)
            data.append(gender_distribution)

            predictions.append(data) # append data to the predictions list

            max_emotionIndex1.append(np.argmax(predictions[0][0]))  # find max of array

            try:
                age01= getAgeRange(int(np.floor(np.sum(predictions[0][1] * output_indexes, axis=1))[0]))
                gender01=getGender(np.argmax(predictions[0][2]))
            except Exception as e:
                print("exception", str(e))

            # when the predictions length is equal 2 get and append another emotion
            if(len(predictions)==2):
                max_emotionIndex2.append(np.argmax(predictions[1][0]))
                cv2.putText(img, emotions[np.argmax(predictions[1][0])], (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255), 2)
            else:
                cv2.putText(img, emotions[np.argmax(predictions[0][0])], (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


            try:
                age02 = getAgeRange(int(np.floor(np.sum(predictions[1][1] * output_indexes, axis=1))[0]))
                gender02 = getGender(np.argmax(predictions[1][2]))
            except Exception as e:
                print("exception", str(e))

            # Update the emotion
            if (shouldUpdateEmotion):
                shouldUpdateEmotion = False  # reset to false
                print("Max emotion 1 Index List: ", max_emotionIndex1)
                print("Max emotion 2 Index List: ", max_emotionIndex2)

                # get the most frequent value in the maxPredictions list
                emotionChoosenIndex1 = getMostFrequentElement(max_emotionIndex1, 6)
                emotionChoosenIndex2 = getMostFrequentElement(max_emotionIndex2, 6)

                emotion01 = emotions[emotionChoosenIndex1] # decide emotion01

                emotion02 = emotions[emotionChoosenIndex2] # decide emotion02

                print("Output Emotion1: ", emotion01)
                print("Output Emotion2: ", emotion02)
                print("Output Gender1: ", gender01)
                print("Output Gender2: ", gender02)
                print("Output Age1: ", age01)
                print("Output Age2: ", age02)

                # reset the emotion when people leave
                if numPep==1:
                    if len(max_emotionIndex2) == 0:
                        emotion02 = "neutral"
                        gender02 = "female"
                        age02 = 'young'
                    indexs =[]
                    indexs_title = df_single_title
                    if(age01=='child'):
                        indexs = list(set(emotionsIndex[emotionChoosenIndex1]).intersection(df_single).intersection(df_child))
                    elif(age01=='young'):
                        indexs = list(set(emotionsIndex[emotionChoosenIndex1]).intersection(df_single).intersection(df_young))
                    elif(age01=='old'):
                        indexs = list(set(emotionsIndex[emotionChoosenIndex1]).intersection(df_single).intersection(df_old))
                else:
                    #indexs = set(emotionsIndex[emotionChoosenIndex1]).intersection(df_multiple)
                    indexs = []
                    indexs_title = df_multiple_title
                    if (age02 == 'child'):
                        indexs = list(
                            set(emotionsIndex[emotionChoosenIndex1]).intersection(df_multiple).intersection(df_child))
                    elif (age02 == 'young'):
                        indexs = list(
                            set(emotionsIndex[emotionChoosenIndex1]).intersection(df_multiple).intersection(df_young))
                    elif (age02 == 'old'):
                        indexs = list(
                            set(emotionsIndex[emotionChoosenIndex1]).intersection(df_multiple).intersection(df_old))

                try:
                    description = get_description_by_index(indexs)
                    title = get_title_by_index(indexs_title)
                except Exception as e:
                    title = random.choice(titles_list)
                    description = random.choice(descriptions_list)

                jsonSender()

                # Reset the List
                max_emotionIndex1.clear()
                max_emotionIndex2.clear()
                max_genderIndex1.clear()
                max_genderIndex2.clear()

            # region Discard Later

            # Draw Emotion Probabilities Table (discarded)
            # for i in range(len(emotions)):
            #     # construct the label text
            #     text = "{}: {:.2f}%".format(emotions[i], predictions[0][i] * 100)
            #     bar_width = int(predictions[0][i] * 300)
            #     cv2.rectangle(canvas, (7, (i * 35) + 5),
            #                   (bar_width, (i * 35) + 35), (0, 0, 255), -1)
            #     cv2.putText(canvas, text, (10, (i * 35) + 23),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            #                 (255, 255, 255), 2)
            # try:
            #     # age gender data set has 40% margin around the face. expand detected face.
            #     margin = 30
            #     margin_x = int((w * margin) / 100)
            #     margin_y = int((h * margin) / 100)
            #     detected_face = img[int(y - margin_y):int(y + h + margin_y), int(x - margin_x):int(x + w + margin_x)]
            # except:
            #     print("detected face has no margin")
            #
            # try:
            #     detected_face = cv2.resize(detected_face, (224, 224))
            #
            #     img_pixels = image.img_to_array(detected_face)
            #     img_pixels = np.expand_dims(img_pixels, axis=0)
            #     img_pixels /= 255
            #     # find out age and gender
            #     age_distributions = age_model.predict(img_pixels)
            #
            #     apparent_age = int(np.floor(np.sum(age_distributions * output_indexes, axis=1))[0])
            #     print("Age: ", apparent_age)
            #
            #     '''
            #     #vgg-face expects inputs (224, 224, 3)
            #     detected_face = cv2.resize(detected_face, (224, 224))
            #
            #     img_pixels = image.img_to_array(detected_face)
            #     img_pixels = np.expand_dims(img_pixels, axis = 0)
            #     img_pixels /= 255
            #
            #     #find out age and gender
            #     age_distributions = age_model.predict(img_pixels)
            #     apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis = 1))[0]))
            #
            #     gender_distribution = gender_model.predict(img_pixels)[0]
            #     gender_index = np.argmax(gender_distribution)
            #
            #     if gender_index == 0: gender = "F"
            #     else: gender = "M"
            #
            #     #background for age gender declaration
            #     info_box_color = (46,200,255)
            #     #triangle_cnt = np.array( [(x+int(w/2), y+10), (x+int(w/2)-25, y-20), (x+int(w/2)+25, y-20)] )
            #     triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
            #     cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
            #     cv2.rectangle(img,(x+int(w/2)-50,y-20),(x+int(w/2)+50,y-90),info_box_color,cv2.FILLED)
            #
            #     #labels for age 
            #     cv2.putText(img, apparent_age, (x+int(w/2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
			#     cv2.putText(img, gender, (x+int(w/2)-42, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
            #     '''
            #
            # except Exception as e:
            #     print("exception", str(e))

            # endregion

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    # cv2.imshow('emotion probabilities', canvas)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('a'):
        jsonSender()

    elif key & 0xFF == ord('q'):  # press q to quit
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
