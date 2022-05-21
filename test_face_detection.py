import cv2
import dlib
import gdown
import argparse

def main(args):

    # The dlib library provides two functions that can be used for face detection:
    #
    #   1. HOG + Linear SVM (fast and less accurate)
    #   2. MMOD CNN face detector (slow and more accurate)

    if args['detector'] == 'hog':
        face_detector = dlib.get_frontal_face_detector()
    else:
        gdown.download(id='1oGNn74w9zU77uEVgzPrLxDG6X8aPzvba', quiet=True)
        face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

    # predict face landmarks
    gdown.download(id='1HChdZjXEIqgqilqU2ar_mMOk-JflK5ah', quiet=True)
    face_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()
        faces = face_detector(image, 1)
        for i, face in enumerate(faces):
            rect = face if args['detector'] == 'hog' else face.rect
            cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
            print(f'face {i}: {rect}')
            shape = face_predictor(image, rect)
            for j in range(0, shape.num_parts):
                (x, y) = (shape.part(j).x, shape.part(j).y)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('Face Landmarks', image)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Landmarks Detection')
    parser.add_argument('-d', '--detector', default='hog', choices=['hog', 'mmod'], help='detector type')
    main(vars(parser.parse_args()))
