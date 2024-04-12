from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.preprocessing import image


app = Flask(__name__)
camera = cv2.VideoCapture(0)
face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# model = load_model("confidence_classifier.h5")


def generate_frames():
    global confidence,con,predictions
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 1)
            for x, y, w, h in faces_detected:
                roi_gray = gray_img[y : y + w, x : x + h]  # cropping region of interest i.e. face area from  image
                roi_gray = cv2.resize(roi_gray, (224, 224))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                # predicting the frames
                # predictions = model.predict(img_pixels)
                # cv2.putText(frame,str(predictions[0][0]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                try:
                    ret, buffer = cv2.imencode(".jpg", frame)
                    frame = buffer.tobytes()
                except :
                    pass
            # con.append(predictions[0][0])
        try:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except:
            pass

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )



if __name__ == "__main__":
    app.run(debug=True)
