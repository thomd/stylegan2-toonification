# Toonification of Real Face Images

In order to toonify real face images we leverage (modified) copies (no git submodules) of several existing repositories:

* https://github.com/justinpinkney/stylegan2 (commit `dbf69a9`) from Justin Pinkney
* https://github.com/dvschultz/stylegan2-ada-pytorch (commit `9b6750b`) from Derrick Schultz

## Setup

    conda env create -f environment.yml
    conda activate toon

## 1. Data Preparation

We use a collection of about 1000 disney/pixar style cartoon face images which were collected using a web
scraper and a custom web tool for image management and image cropping.

Store cartoon face images in `./cartoon-images`.

### Cartoon Face Alignment

As we are going to apply transfer-learning on the **FFHQ_1024 model**, we have to **resize** all images to 1024&times;1024 and **align** all 
cartoon images to have similar face-keypoint positions as the face images from the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) used to train the FFHQ model.

In oder to find face-keypoints, we need to begin with detecting cartoon faces.

The [dlib](http://dlib.net/) library provides two functions that can be used for cartoon face detection:

1. **HOG + Linear SVM** face detection (fast and less accurate)
2. **MMOD CNN** face detector (slow and more accurate)

We detect 68 face landmarks using the **pretrained landmarks detector** model [shape_predictor_68_face_landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) from dlib.

To get an idea what landmark detection means, run

    > python test_face_detection.py --detector hog
    > python test_face_detection.py --detector mmod

For aligining the cartoon images similar to FFHQ, we leverage the `face_alignment.py` script from the `stylegan2` repository:

    > python align_image_data.py --help
    > python align_image_data.py --images-source /path/to/images --images-target dataset

The `align_image_data.py` script uses the MMOD face detection model by default, as it detects more cartoon faces. 

In order to **visually verify** and **cleanup** properly aligned cartoon faces (the `remove` button deletes images from the target folder), open a Jupyter notebook and run

    from utils import print_images
    print_images(image_filter='aligned', size=512, landmarks=True)

To visually verify which cartoon face could not be aligned or which were deleted manually, run

    from utils import print_images
    print_images(image_filter='missed')

## 2. Finetuning FFHQ Model

The repository `stylegan2-ada-pytorch` allows us to finetune the pre-trained **FFHQ model** using
transfer-learning.

This model was created using [StyleGAN2](https://github.com/NVlabs/stylegan2), which is an improved **generative adversarial network** (GAN) published by Nvidia 2020.

StyleGAN2-ADA is a further improved GAN which leverages **adaptive discriminator augmentation** (ADA) to prevent overfitting due to a small dataset.

StyleGAN2-ADA requires a GPU which is available using [Google Colab](https://colab.research.google.com):

### Setup on Google Colab

first, zip the images dataset and upload to a Google Drive folder:

    > gdrive mkdir toonification
    > gdrive upload -p <folderId> dataset.zip

Good practice is to store all training results on Google Drive. Mount Google Drive and create a project folder with

    from google.colab import drive
    drive.mount('/content/drive')
    project = '/content/drive/MyDrive/toonification'
    %mkdir -p {project}

Then install dependencies with

    !git clone https://github.com/thomd/stylegan2-toonification.git
    !pip install --quiet opensimplex ninja
    !nvidi-smi
    %cd stylegan2-toonification/stylegan2-ada-pytorch/



