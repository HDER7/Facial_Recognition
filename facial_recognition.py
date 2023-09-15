import cv2
import face_recognition as fr

# Update images
control_photo = fr.load_image_file('employs/Ye.jpg')
test_photo = fr.load_image_file('230730113612-kanye-051323.jpg')

# Converter images to RGB
control_photo = cv2.cvtColor(control_photo, cv2.COLOR_BGR2RGB)
test_photo = cv2.cvtColor(test_photo, cv2.COLOR_BGR2RGB)

# Localize control face
face = fr.face_locations(control_photo)[0]
code_face = fr.face_encodings(control_photo)[0]
print(face)

# Localize test face
test_face = fr.face_locations(test_photo)[0]
test_code_face = fr.face_encodings(test_photo)[0]
print(test_face)

# Show rectangles
cv2.rectangle(control_photo, (face[3], face[0]),
              (face[1], face[2]),
              (0, 255, 0),
              2)

cv2.rectangle(test_photo, (test_face[3], test_face[0]),
              (test_face[1], test_face[2]),
              (0, 255, 0),
              2)

# Compare Faces
compare = fr.compare_faces([code_face], test_code_face)
print(compare)

# Distance measure
distance = fr.face_distance([code_face], test_code_face)
print(distance)

# Show result
cv2.putText(test_photo,
            f'{compare} {distance.round(2)}',
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2)

# Show images
cv2.imshow('Control Photo', control_photo)
cv2.imshow('Test Photo', test_photo)

# Keep open the program
cv2.waitKey(0)
