# Toonification of Real Face Images

In order to toonify real face images, we use (modified) copies (no git submodules) of several repositories:

* https://github.com/justinpinkney/stylegan2

## Setup

    conda env create -f environment.yml
    conda activate toon

## Data Preparation

We use a collection of about 1000 disney/pixar style cartoon face images which were collected using a web
scraper and a custom web tool for image management and image cropping.

### Face Alignment

As we are going to apply transfer-learning on the **FFHQ_512 model**, we have to **resize** all images to 512&times;512 and **align** all 
cartoon images to have similar face-keypoint positions as the face images used to train FFHQ.

In oder to find face-keypoints, we need to begin with detectng faces. 

The [dlib](http://dlib.net/) library provides two functions that can be used for face detection:

1. **HOG + Linear SVM** face detection (fast and less accurate)
2. **MMOD CNN** face detector (slow and more accurate)

We detect 68 face landmarks using the **pretrained landmarks detector** model [shape_predictor_68_face_landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) from dlib.

To get an idea what landmark detection means, run

    python test_face_detection.py --detector hog
    python test_face_detection.py --detector mmod


