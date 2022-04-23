# Toonification of real face images

In order to toonify real face images, we use (modified) copies (no git submodules) of several repositories:

* https://github.com/justinpinkney/stylegan2

## Data Preparation

We use a collection of about 1000 disney/pixar style cartoon face images which were collected using a web
scraper and a custom web tool for image management and image cropping.

As we are going to apply transfer-learning on the **FFHQ_512 model**, we have to **resize** all images to 512&times;512 and **align** all cartoon images to have the
same landmarks (face-keypoints):

    python 
