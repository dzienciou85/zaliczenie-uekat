# import OpenCV
import cv2
# import Flask and Flas restful modules
from flask import Flask, request
from flask_restful import Resource, Api
import urllib.request
# import requests
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


class PeopleCountStatic(Resource):
    def get(self):
        image = cv2.imread('pobrane.jpg')
        image = cv2.resize(image, (600, 450))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(1, 1), padding=(7, 7), scale=1.04)
        return {'People on image': len(rects)}


class PeopleDynamicUrl(Resource):
    def get(self):
        url = request.args.get('url')
        urllib.request.urlretrieve(url, "obrazekzneta.jpg")
        # img = Image.open(r"obrazekzneta.jpg")
        img = cv2.imread("obrazekzneta.jpg")
        # img = cv2.resize(img, (600, 450))
        # resizing with proportions
        img = cv2.resize(img, None, fx=0.85, fy=0.85)

        # detect people in the image
        (rects2, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(10, 10), scale=1.1)
        return {'People found in URL:': len(rects2)}
        pass


api.add_resource(PeopleCountStatic, '/')
api.add_resource(PeopleDynamicUrl, '/dynamic')

if __name__ == '__main__':
    app.run(debug=True)
