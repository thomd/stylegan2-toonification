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
    print_images(source_folder='/path/to/images', target_folder='dataset', image_filter='aligned', landmarks=True)

To visually verify which cartoon face could not be aligned or which were deleted manually, run

    from utils import print_images
    print_images(source_folder='/path/to/images', target_folder='dataset', image_filter='missed')

## 2. Finetuning FFHQ Model

The repository `stylegan2-ada-pytorch` allows us to finetune the pre-trained **FFHQ model** using
transfer-learning.

This model was created using [StyleGAN2](https://github.com/NVlabs/stylegan2), which is an improved **generative adversarial network** (GAN) published by Nvidia 2020.

StyleGAN2-ADA is a further improved GAN which leverages **adaptive discriminator augmentation** (ADA) to prevent overfitting due to a small dataset.

StyleGAN2-ADA requires a GPU which is available using [Google Colab](https://colab.research.google.com):

### Setup on Google Colab

Good practice is to store all datasets and training results on **Google Drive**.

First, mount Google Drive and create a project folder with

    from google.colab import drive
    drive.mount('/content/drive')
    project = '/content/drive/MyDrive/toonification'
    %mkdir -p {project}

then, zip the images dataset and upload to a Google Drive folder:

    > gdrive list -q "name contains 'toonification' and mimeType contains 'folder'"
    > gdrive upload -p <folderId> dataset.zip

then install dependencies with

    !git clone https://github.com/thomd/stylegan2-toonification.git
    !pip install --quiet ninja opensimplex torch==1.7.1 torchvision==0.8.2
    !nvidi-smi
    %cd stylegan2-toonification/stylegan2-ada-pytorch/

Start **Tensorboard** for tracking metrics (you might Firefox deactivate "Enhanced Tracking Protection" in Firefox):

    %load_ext tensorboard
    %tensorboard --logdir={project}/results/

### Resume Training of Pre-Trained FFHQ Model

Google Colab notebooks have an idle timeout of 90 minutes and absolute timeout of 12 hours. This means, if user does not interact with his Google Colab notebook for more than 90 minutes, its instance is automatically terminated. Also, maximum lifetime of a Colab instance is 12 hours.

Therefore it is good practice to save a snapshop very often (setting `snap = 1`) and resume with the last snapshot after 12 hours.

Start fine-tuning the FFHQ model:

    resume = 'ffhq1024'
    nkimg = 0
    aug = 'ada'
    augpipe = 'bgcf'
    freezed = 0
    gamma = 10
    target = 0.6

    !python train.py \
            --cfg='11gb-gpu' \
            --outdir={project}/results \
            --data=/content/drive/MyDrive/toonification/dataset.zip \
            --snap=1 \
            --resume={resume} \
            --freezed={freezed} \
            --aug={aug} \
            --augpipe={augpipe} \
            --gamma={gamma} \
            --target={target} \
            --mirror=True \
            --nkimg={nkimg}

When resuming with a sanpshot, set `resume` and `nkimg` accordingly and start the trainign script again:

    resume = {project}/results/.../network-snapshot-000xxx.pkl
    nkimg = xxx

Stop trainign as soon as the losses in Tensorboard reach a plateau. Then do variations of `freezed`, `gamma`, `aug`,
`augpip` and `target` and keep results for later tests with real face images.





