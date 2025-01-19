import os
import zipfile

local_zip = r'E:\Visual Studio Code\Facultate\inteligenta artificiala\horse-or-human'


if os.path.exists(local_zip):
    try:
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall('/tmp/horse-or-human')
        zip_ref.close()
        print('Fișierul ZIP a fost extras cu succes.')
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
else:
    print(f"Fișierul ZIP nu există la calea: {local_zip}")

train_horse_dir = r'E:\Visual Studio Code\Facultate\inteligenta artificiala\horse-or-human\horses'


train_human_dir = r'E:\Visual Studio Code\Facultate\inteligenta artificiala\horse-or-human\humans'


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
 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


ncols = 4
nrows = 5
pic_index = 0  # Indexul imaginilor care urmează să fie afișate


fig = plt.gcf()
fig.set_size_inches(ncols * 5, nrows * 5)


pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) 
                  for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) 
                  for fname in train_human_names[pic_index-8:pic_index]]


if len(next_horse_pix) + len(next_human_pix) > ncols * nrows:
    print("Prea multe imagini pentru dimensiunile alese!")
else:
    for i, img_path in enumerate(next_horse_pix + next_human_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        try:
            img = mpimg.imread(img_path)
            plt.imshow(img)
        except FileNotFoundError:
            print(f"Fișierul nu a fost găsit: {img_path}")

    # Afișează toate imaginile
    plt.show()
    

