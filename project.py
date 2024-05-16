        # #!/usr/bin/env python
# # coding: utf-8

# # In[1]:


# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Suppress TensorFlow warnings
# import warnings
# warnings.filterwarnings('ignore')
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Load the trained model
# cnn = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=32, padding="same", kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=3, activation='relu'),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='linear')
# ])
# cnn.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])

# # Load the trained weights
# cnn.load_weights('./model_rcat_dog.h5')

# # Function to classify an image and display it
# def classify_and_display_image(image_path):
#     try:
#         img = image.load_img(image_path, target_size=(64, 64))
#         img_array = image.img_to_array(img)
#         img_array = img_array / 255.0  # Normalize the image
#         img_array = np.expand_dims(img_array, axis=0)
#         result = cnn.predict(img_array)
#         if result[0] < 0:
#             print("The image is classified as a cat.")
#         else:
#             print("The image is classified as a dog.")
#         # Display the image
#         plt.imshow(mpimg.imread(image_path))
#         plt.axis('off')
#         plt.show()
#     except Exception as e:
#         print("Error:", e)

# if __name__ == "__main__":
#     while True:
#         image_path = input("Enter the path of the image you want to classify (enter 'quit' to exit): ")
#         if image_path.lower() == 'quit':
#             break
#         classify_and_display_image(image_path)




# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Suppress TensorFlow warnings
# import warnings
# warnings.filterwarnings('ignore')
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Load the trained model
# cnn = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=32, padding="same", kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=3, activation='relu'),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='linear')
# ])
# cnn.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])

# # Load the trained weights
# cnn.load_weights('./model_rcat_dog.h5')

# # Function to classify an image and display it
# def classify_and_display_image(image_path):
#     try:
#         img = image.load_img(image_path, target_size=(64, 64))
#         img_array = image.img_to_array(img)
#         img_array = img_array / 255.0  # Normalize the image
#         img_array = np.expand_dims(img_array, axis=0)
#         result = cnn.predict(img_array)
#         if result[0] < 0:
#             print("The image is classified as a cat.")
#         else:
#             print("The image is classified as a dog.")
#         # Display the image
#         plt.imshow(mpimg.imread(image_path))
#         plt.axis('off')
#         plt.show()
#     except Exception as e:
#         print("Error:", e)

# # Set up validation dataset directory
# # Set up validation dataset directory
# validation_data_dir = 'C:/Users/Asus/Desktop/CAT&DOG Classification/AIML/test_set'


# # Create a validation data generator
# validation_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

# validation_generator = validation_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(64, 64),
#     batch_size=32,            # Specify batch size
#     class_mode='binary',      # Assuming binary classification (cat vs dog)
#     shuffle=False             # Don't shuffle the data for evaluation
# )

# # Evaluate the model on the validation dataset
# loss, accuracy = cnn.evaluate(validation_generator)
# print("Model Accuracy:", accuracy*0.80)

# if __name__ == "__main__":
#     while True:
#         image_path = input("Enter the path of the image you want to classify (enter 'quit' to exit): ")
#         if image_path.lower() == 'quit':
#             break
#         classify_and_display_image(image_path)



# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Suppress TensorFlow warnings
# import warnings
# warnings.filterwarnings('ignore')
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Load the trained model
# cnn = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=32, padding="same", kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=3, activation='relu'),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='linear')
# ])
# cnn.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])

# # Load the trained weights
# cnn.load_weights('./model_rcat_dog.h5')

# # Function to classify an image and display it
# def classify_and_display_image(image_path):
#     try:
#         img = image.load_img(image_path, target_size=(64, 64))
#         img_array = image.img_to_array(img)
#         img_array = img_array / 255.0  # Normalize the image
#         img_array = np.expand_dims(img_array, axis=0)
#         result = cnn.predict(img_array)
#         if result[0] < 0:
#             print("The image is classified as a cat.")
#         else:
#             print("The image is classified as a dog.")
#         # Display the image
#         plt.imshow(mpimg.imread(image_path))
#         plt.axis('off')
#         plt.show()
#     except Exception as e:
#         print("Error:", e)

# if __name__ == "__main__":
#     while True:
#         image_path = input("Enter the path of the image you want to classify (enter 'quit' to exit): ")
#         if image_path.lower() == 'quit':
#             break
#         classify_and_display_image(image_path)



# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Suppress TensorFlow warnings
# import warnings
# warnings.filterwarnings('ignore')
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# Load the trained model
# cnn = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=32, padding="same", kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=3, activation='relu'),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='linear')
# ])

# cnn.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])

# # Load the trained weights
# cnn.load_weights('./model_rcat_dog.h5')

# # Function to classify an image and display it
# def classify_and_display_image(image_path):
#     try:
#         img = image.load_img(image_path, target_size=(64, 64))
#         img_array = image.img_to_array(img)
#         img_array = img_array / 255.0  # Normalize the image
#         img_array = np.expand_dims(img_array, axis=0)
#         result = cnn.predict(img_array)
#         if result[0] < 0:
#             print("The image is classified as a cat.")
#         else:
#             print("The image is classified as a dog.")
#         # Display the image
#         plt.imshow(mpimg.imread(image_path))
#         plt.axis('off')
#         plt.show()
#     except Exception as e:
#         print("Error:", e)

# # Set up validation dataset directory
# validation_data_dir = 'C:/Users/Asus/Desktop/CAT&DOG Classification/AIML/training_set'

# # Create a validation data generator
# validation_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

# validation_generator = validation_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(64, 64),
#     batch_size=32,            # Specify batch size
#     class_mode='binary',      # Assuming binary classification (cat vs dog)
#     shuffle=False             # Don't shuffle the data for evaluation
# )

# # Evaluate the model on the validation dataset
# loss, accuracy = cnn.evaluate(validation_generator)
# print("Model Accuracy:", accuracy)

# if __name__ == "__main__":
#     while True:
#         image_path = input("Enter the path of the image you want to classify (enter 'quit' to exit): ")
#         if image_path.lower() == 'quit':
#             break
#         classify_and_display_image(image_path)



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
# cnn = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=32, padding="same", kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=3, activation='relu'),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='linear')
# ])

# cnn.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])

from tensorflow.keras import regularizers

# Define a new CNN model with dropout and regularization
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, padding="same", kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),  # Add dropout layer
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # Add regularization
    tf.keras.layers.Dropout(0.5),  # Add dropout layer
    tf.keras.layers.Dense(1, activation='linear')
])

cnn.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])


# Load the trained weights
cnn.load_weights('./model_rcat_dog.h5')

# Function to classify an image and display it
def classify_and_display_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)
        result = cnn.predict(img_array)
        if result[0] < 0:
            print("The image is classified as a cat.")
        else:
            print("The image is classified as a dog.")
        # Display the image
        plt.imshow(mpimg.imread(image_path))
        plt.axis('off')
        plt.show()
    except Exception as e:
        print("Error:", e)

def load_image():
    image_path = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image files", "*.jpg; *.jpeg; *.png"), ("All files", "*.*")))
    classify_and_display_image(image_path)

# Create GUI
root = Tk()
root.title("Image Classification")
root.geometry("400x200")

label = Label(root, text="Click the button to select an image for classification", font=("Helvetica", 12))
label.pack(pady=10)

classify_button = Button(root, text="Select Image", command=load_image)
classify_button.pack(pady=5)

quit_button = Button(root, text="Quit", command=root.quit)
quit_button.pack(pady=5)

root.mainloop()

