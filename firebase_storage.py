import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import cv2


class Firestore:
    def __init__(self):
        cred = credentials.Certificate("key_smart_home.json")
        firebase_admin.initialize_app(cred, {'storageBucket': 'smarthome-b09fa.appspot.com'})
        self.bucket = storage.bucket()

    def get_image(self):
        blobs = self.bucket.list_blobs()
        for b in blobs:
            if b.size is not None:
                path = b.name
                print(path)

        blob = self.bucket.blob(path)
        img_arr = np.frombuffer(blob.download_as_string(), np.uint8)
        img = cv2.imdecode(img_arr, cv2.COLOR_BGR2BGR555)
        return img


images = Firestore().get_image()
if images is not None:
    cv2.imshow("images", images)
    cv2.waitKey(0)
