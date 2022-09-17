import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#model = models.load_model('best_model.h5')
#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
video = cv2.VideoCapture(-1)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        #im = Image.fromarray(frame, 'RGB')

        #Resizing into dimensions you used while training
        #im = im.resize((200,200,1))
        #img_array = np.array(im)

        #Expand dimensions to match the 4D Tensor shape.
        #img_array = np.expand_dims(img_array, axis=0)
        #img_array = np.reshape(100,)

        #Calling the predict function using keras
        #prediction = model.predict(img_array)#[0][0]
        #print(prediction)
        print("hello")
        #Customize this part to your liking...
        '''
        if(prediction == 1 or prediction == 0):
            print("No Human")
        elif(prediction < 0.5 and prediction != 0):
            print("Female")
        elif(prediction > 0.5 and prediction != 1):
            print("Male")
            '''

        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()
