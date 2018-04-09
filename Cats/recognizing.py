import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import argparse as ar

parser = ar.ArgumentParser()
parser.add_argument('path', metavar='path', type=str, nargs='+',
                    help='an integer for the accumulator')

img_path = parser.parse_args().path[0]
print('path = ' + img_path)
print('start loading image...')
img = image.load_img(img_path, target_size=(150, 150))
print('start creating array from image...')
x = image.img_to_array(img)
print('start x /= 255...')
x /= 255
print('start expanding x...')
x = np.expand_dims(x, axis=0)
print('openning json...')
json_file = open("/home/shamil/CatsAndroid/cats_vgg19_model.json", "r")
print('reading json...')
loaded_model_json = json_file.read()
print('closing json...')
json_file.close()
print('creating model from json...')
loaded_model = model_from_json(loaded_model_json)
print('loading weights from cats_vgg19_model.h5...')
loaded_model.load_weights("/home/shamil/CatsAndroid/cats_vgg19_model.h5")
print('compiling weights from cats_vgg19_model.h5...')
loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print('getting model prediction from x...')
prediction = loaded_model.predict(x)
print('creating array of breeds...')
classes = ['Абиссинская', 'Американская короткошерстная', 'Бенгальская', 'Бирманская', 'Бомбейская',
           'Британская короткошерстная', 'Египетский мау', 'Мейн кун', 'Персидская', 'Рэгдолл', 'Русская голубая',
           'Сиамская', 'Сфинкс']
print(classes[np.argmax(prediction)])