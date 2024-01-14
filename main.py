# import OpenCV
import cv2
# import Flask and Flas restful modules
from flask import Flask
from flask_restful import Resource, Api
# Read the image and checking its type
img = cv2.imread('pobrane.jpg')
img = cv2.resize(img, (600, 450))

print(type(img))
print(img.shape)



# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)
api = Api(app)


class PeopleCount(Resource):
    def get(self):
        image = cv2.imread('pobrane.jpg')
        image = cv2.resize(image, (600, 450))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(1, 1), padding=(8, 8), scale=1.04)
        return {'People on image': len(rects)}


api.add_resource(PeopleCount, '/')

if __name__ == '__main__':
    app.run(debug=False)

