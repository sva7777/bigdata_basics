import cv2

from IPython.display import display, clear_output

from matplotlib import pyplot as plt

smile = cv2.imread('./Data/smile.jpg')
no_smile = cv2.imread('./Data/nosmile.jpg')

# visualize the two test images
fig, ax = plt.subplots(1, 2)
clear_output()
ax[0].imshow(cv2.cvtColor(smile, cv2.COLOR_BGR2GRAY), cmap='gray')
ax[0].axis('off')
ax[0].set_title('Smile')
ax[1].imshow(cv2.cvtColor(no_smile, cv2.COLOR_BGR2GRAY), cmap='gray')
ax[1].axis('off')
ax[1].set_title('No smile')

plt.show()

# path to the file containing the features that the openCv pipeline will look for in the frame
cascadePath = "./Data/haarcascade_frontalface_default.xml"


# initialize a model for detecting whether a certain portion of an image contains a face

faceCascade = cv2.CascadeClassifier(cascadePath)

gray_smile = cv2.cvtColor(smile, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
        gray_smile,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
# output of the classifier
print( faces)



from matplotlib.patches import Rectangle

# visualize the detected face
fig, ax = plt.subplots()
ax.imshow(gray_smile, cmap='gray')

# iterate over all the detected faces
for face in faces:
    # retrieve the coordinates of the position of the current face, and its size
    (x_smile, y_smile, w_smile, h_smile) = face
    # draw a rectangle where the face is detected
    ax.add_artist(Rectangle((x_smile, y_smile), w_smile, h_smile, fill=False, lw=3, color='green'))

ax.axis('off')

plt.show()


gray_smile = cv2.cvtColor(smile, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
        gray_smile,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
# output of the classifier
print( faces)

from matplotlib.patches import Rectangle


##
gray_no_smile = cv2.cvtColor(no_smile, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
        gray_no_smile,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
# output of the classifier
print( faces)


# visualize the detected face
fig, ax = plt.subplots()
ax.imshow(gray_no_smile, cmap='gray')

# iterate over all the detected faces
for face in faces:
    # retrieve the coordinates of the position of the current face, and its size
    (x_no_smile, y_no_smile, w_no_smile, h_no_smile) = face
    # draw a rectangle where the face is detected
    ax.add_artist(Rectangle((x_no_smile, y_no_smile), w_no_smile, h_no_smile, fill=False, lw=3, color='green'))

ax.axis('off')

plt.show()






# select only the face portion from the smile test image
face_smile = gray_smile[y_smile:y_smile+h_smile, x_smile:x_smile+w_smile]

# The cropping coefficient for determining the size of the face
c1 = 0.2

# calculate how to crop the face
# vertical dimension
v_cut = int(c1 * w_smile)
# horizontal dimension
h_cut = int(c1 * h_smile)

# select only the face portion from the smile test image
cut_face_smile = gray_smile[y_smile+v_cut:y_smile+h_smile,
                      x_smile+h_cut:x_smile-h_cut+w_smile]



# select only the face portion from the no smile test image
face_no_smile = gray_no_smile[y_no_smile:y_no_smile+h_no_smile, x_no_smile:x_no_smile+w_no_smile]

# The cropping coefficient for determining the size of the face
c1 = 0.2

# calculate how to crop the face
# vertical dimension
v_cut = int(c1 * w_no_smile)
# horizontal dimension
h_cut = int(c1 * h_no_smile)

# select only the face portion from the smile test image
cut_face_no_smile = gray_no_smile[y_no_smile+v_cut:y_no_smile+h_no_smile,
                      x_no_smile+h_cut:x_no_smile-h_cut+w_no_smile]

fig, ax = plt.subplots(2, 2)
ax[0][0].imshow(face_smile, cmap='gray')
ax[0][0].axis('off')
ax[0][0].set_title('Original')

ax[0][1].imshow(cut_face_smile, cmap='gray')
ax[0][1].axis('off')
ax[0][1].set_title('Cropped')

ax[1][0].imshow(face_no_smile, cmap='gray')
ax[1][0].axis('off')
ax[1][1].imshow(cut_face_no_smile, cmap='gray')
ax[1][1].axis('off')

plt.show()

import numpy as np
from scipy.ndimage import zoom

# transform the stretched smiling face so that it has 64x64 pixels
standardized_face_smile = zoom(cut_face_smile, (64. / cut_face_smile.shape[0],
                                           64. / cut_face_smile.shape[1])).astype(np.float32)

# normalize the image so that its values are between 0 and 1
standardized_face_smile /= float(255)

# transform the stretched no smiling face so that it has 64x64 pixels
standardized_face_no_smile = zoom(cut_face_no_smile, (64. / cut_face_no_smile.shape[0],
                                           64. / cut_face_no_smile.shape[1])).astype(np.float32)

# normalize the image so that its values are between 0 and 1
standardized_face_no_smile /= float(255)

plt.subplot(121)
plt.imshow(standardized_face_smile[:, :], cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(standardized_face_no_smile[:, :], cmap='gray')
plt.axis('off')

plt.show()

import pickle as pkl


# load LR model
with open('./Data/pl-app_2.0.0-support_vector_machines.pkl', 'rb') as f:
    classifier = pkl.load(f)

