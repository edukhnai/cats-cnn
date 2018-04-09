from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import VGG19
from PIL import ImageFile

# Каталог с данными для обучения
train_dir = 'train1'
# Каталог с данными для проверки
val_dir = 'validate1'
# Каталог с данными для тестирования
test_dir = 'test1'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 10
# Размер мини-выборки
batch_size = 64
# Количество изображений для обучения
nb_train_samples = 13650
# Количество изображений для проверки
nb_validation_samples = 2925
# Количество изображений для тестирования
nb_test_samples = 2925

vgg19_net = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

vgg19_net.trainable = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

model = Sequential()
# Добавляем в модель сеть VGG19 вместо слоя
model.add(vgg19_net)
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(13))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

print("Точность на тестовых данных: %.2f%%" % (scores[1]*100))

# Генерируем описание модели в формате json
model_json = model.to_json()
# Записываем модель в файл
json_file = open("cats_vgg19_model.json", "w")
json_file.write(model_json)
json_file.close()

model.save_weights("cats_vgg19_model.h5")