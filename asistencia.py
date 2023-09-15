import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime

# Create BD
rute = 'employs'
images = []
employee_name = []
employs_list = os.listdir(rute)

for nom in employs_list:
    actual_imagen = cv2.imread(f'{rute}\\{nom}')
    images.append(actual_imagen)
    employee_name.append(os.path.splitext(nom)[0])


# Encode images
def encode(img):
    encode_list = []
    for i in img:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        code = fr.face_encodings(i)[0]
        encode_list.append(code)
    return encode_list


# Income employee
def register(person):
    f = open('register.csv', 'r+')
    data = f.readlines()
    noms = []
    for line in data:
        income = line.split(',')
        noms.append(income[0])
    if person not in noms:
        now = datetime.now()
        st = now.strftime('%H:%M:%S')
        f.writelines(f'\n{person}, {st}')


encode_employee_list = encode(images)

# Camera
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Read Camera
success, image = capture.read()
if not success:
    print("No se ha reconocido")
else:
    face_capture = fr.face_locations(image)
    face_encode = fr.face_encodings(image, face_capture)
    # Auth
    for en_face, face_c in zip(face_encode, face_capture):
        coincidences = fr.compare_faces(encode_employee_list, en_face)
        distances = fr.face_distance(encode_employee_list, en_face)

        coincidence_index = numpy.argmin(distances)
        if distances[coincidence_index] > 0.6:
            print('No coincide con ninguno de los empleados')
        else:
            nom = employee_name[coincidence_index]
            y1, x2, y2, x1 = face_c
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), 2, cv2.FILLED)
            cv2.putText(image, nom, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        2)
            register(nom)
            cv2.imshow('imagen', image)
            cv2.waitKey(0)
