import cv2
from flask import Flask
from flask_restful import Resource, Api
img = cv2.imread('pobrane.jpg')
img = cv2.resize(img, (600, 450))

print(type(img))
print(img.shape)

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# # load image
# image = cv2.imread('pobrane.jpg')
# image = cv2.resize(image, (600, 450))
#
# # detect people in the image
# (rects, weights) = hog.detectMultiScale(image, winStride=(1, 1), padding=(8, 8), scale=1.04)
#
# # draw the bounding boxes
# for (x, y, w, h) in rects:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# print(f'Found {len(rects)} humans')

# show the output images
# cv2.imshow("People detector", image)
# cv2.waitKey(0)

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
