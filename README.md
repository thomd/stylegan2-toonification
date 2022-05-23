# Toonification of Real Face Images

In order to toonify real face images, we use (modified) copies (no git submodules) of several repositories:

* https://github.com/justinpinkney/stylegan2

## Setup

    conda env create -f environment.yml
    conda activate toon

## Data Preparation

We use a collection of about 1000 disney/pixar style cartoon face images which were collected using a web
scraper and a custom web tool for image management and image cropping.

Store cartoon face images in `./cartoon-images`.

### Cartoon Face Alignment

As we are going to apply transfer-learning on the **FFHQ_512 model**, we have to **resize** all images to 512&times;512 and **align** all 
cartoon images to have similar face-keypoint positions as the face images used to train FFHQ.

In oder to find face-keypoints, we need to begin with detecting cartoon faces.

The [dlib](http://dlib.net/) library provides two functions that can be used for cartoon face detection:

1. **HOG + Linear SVM** face detection (fast and less accurate)
2. **MMOD CNN** face detector (slow and more accurate)

We detect 68 face landmarks using the **pretrained landmarks detector** model [shape_predictor_68_face_landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) from dlib.

To get an idea what landmark detection means, run

    python test_face_detection.py --detector hog
    python test_face_detection.py --detector mmod

For aligining the cartoon images similar to FFHQ, we leverage the `face_alignment.py` script from the `stylegan2` repository:

    python align_image_data.py --help
    python align_image_data.py

The `align_image_data.py` script uses the MMOD face detection model by default, as it detects more cartoon faces. 

In order to **visually verify** and **cleanup** properly aligned cartoon faces (the `remove` button deletes images from the target folder), open a Jupyter notebook and run

    from utils import print_images
    print_images(image_filter='aligned', size=512, landmarks=True)

To visually verify which cartoon face could not be aligned or which were deleted manually, run

    from utils import print_images
    print_images(image_filter='missed')

