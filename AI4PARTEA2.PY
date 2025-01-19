import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Calea pentru fișierul ZIP
local_zip = r'E:\Visual Studio Code\Facultate\inteligenta artificiala\horse-or-human'

if os.path.exists(local_zip):
    try:
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall(r'E:\Visual Studio Code\Facultate\inteligenta artificiala\horse-or-human')
        zip_ref.close()
        print('Fișierul ZIP a fost extras cu succes.')
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
else:
    print(f"Fișierul ZIP nu există la calea: {local_zip}")

train_horse_dir = r'E:\Visual Studio Code\Facultate\inteligenta artificiala\horse-or-human\horses'
train_human_dir = r'E:\Visual Studio Code\Facultate\inteligenta artificiala\horse-or-human\humans'

# Verifică existența directoarelor și afișează fișierele
if os.path.isdir(train_horse_dir):
    train_horse_names = os.listdir(train_horse_dir)
    print(train_horse_names[:10])
else:
    print('Directorul pentru pozele de cal nu există.')

if os.path.isdir(train_human_dir):
    train_human_names = os.listdir(train_human_dir)
    print(train_human_names[:10])
else:
    print('Directorul pentru pozele de om nu există.')

print('total training horse images:', len(os.listdir(train_horse_dir)) if os.path.isdir(train_horse_dir) else 0)
print('total training human images:', len(os.listdir(train_human_dir)) if os.path.isdir(train_human_dir) else 0)


ncols = 4
nrows = 5
pic_index = 0
fig = plt.gcf()
fig.set_size_inches(ncols * 5, nrows * 5)
pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix + next_human_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')
    try:
        img = mpimg.imread(img_path)
        plt.imshow(img)
    except FileNotFoundError:
        print(f"Fișierul nu a fost găsit: {img_path}")

plt.show()

# Definirea modelului CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])
model.summary()

# Preprocesarea imaginilor
train_datagen = ImageDataGenerator(rescale=1./255)

# Încărcarea imaginilor din directorul corect
train_generator = train_datagen.flow_from_directory(
    r'E:\Visual Studio Code\Facultate\inteligenta artificiala\horse-or-humann',  
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1)