import cv2, os
import numpy as np
from PIL import Image


class DetectFace:
    def __init__(self):
        cascadePath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascadePath)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.images = []
        self.labels = []

    def check_and_get_face(self, image_path: str):
        # Переводим изображение в черно-белый формат и приводим его к формату массива
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        # Из каждого имени файла извлекаем номер человека, изображенного на фото
        subject_number = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        # Определяем области где есть лица
        faces = self.faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Если лицо нашлось добавляем его в список images, а соответствующий ему номер в список labels
        for (x, y, w, h) in faces:
            self.images.append(image[y: y + h, x: x + w])
            self.labels.append(subject_number)

    def get_images(self, path: str = 'img'):
        # Ищем все фотографии и записываем их в image_paths
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]

        for image_path in image_paths:
            self.check_and_get_face(image_path)

    def response_img(self, path: str = 'img/subject01.surprised'):
        # Ищем лица на фотографиях
        gray = Image.open(path).convert('L')
        image = np.array(gray, 'uint8')
        faces = self.faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Если лица найдены, пытаемся распознать их
            # Функция  recognizer.predict в случае успешного распознавания возвращает номер и параметр confidence,
            # этот параметр указывает на уверенность алгоритма, что это именно тот человек, чем он меньше, тем больше уверенность
            number_predicted, conf = self.recognizer.predict(image[y: y + h, x: x + w])
            # Извлекаем настоящий номер человека на фото и сравниваем с тем, что выдал алгоритм
            number_actual = int(os.path.split(path)[1].split(".")[0].replace("subject", ""))
            if number_actual == number_predicted:
                return True
            return False

    def init_image(self):
        # Получаем лица и соответствующие им номера
        self.get_images()
        # Обучаем программу распознавать лица
        self.recognizer.train(self.images, np.array(self.labels))


if __name__ == '__main__':
    a = DetectFace()
    a.init_image()
    a.response_img()
    a.response_img('/home/sima/Documents/course/img/subject03.sleepy')
