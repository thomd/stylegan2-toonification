import numpy as np
from pathlib import Path
from PIL import Image
import ipywidgets as widgets
from IPython.display import display
import cv2
import dlib
import pickle


def remove_image(btn):
    print(f'removed image {btn.image_path}')
    btn.style.button_color = 'LightCoral'
    btn.description = 'removed'
    Path(btn.image_path).unlink()


def remove_button(file_path):
    button = widgets.Button(description='remove', button_style='')
    button.on_click(remove_image)
    button.image_path = file_path
    return button


def print_images(source_folder='faces', target_folder='faces-aligned', image_filter='aligned', landmarks=False, size=512, **kwargs):
    image_types = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    source_images = [str(p) for p in Path(source_folder).glob('**/*.*') if p.suffix in image_types]
    if 'image_count' in kwargs:
        source_images = source_images[:kwargs['image_count']]

    for i, image_path in enumerate(source_images):
        image = dlib.load_rgb_image(image_path)

        if landmarks:
            landmarks_file = Path(target_folder + '/' + str(Path(image_path).stem) + '_00.pkl')
            if landmarks_file.exists():
                infile = open(landmarks_file, 'rb')
                rect, landmarks = pickle.load(infile)
                infile.close()
                cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
                for landmark in landmarks:
                    cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)
            img1 = image
        else:
            img1 = np.array(Image.fromarray(image).resize((size, size)))

        aligned_image = Path(target_folder + '/' + str(Path(image_path).stem) + '_00.png')

        if aligned_image.exists():
            image = dlib.load_rgb_image(str(aligned_image))
            img2 = np.array(Image.fromarray(image).resize((size, size)))
        else:
            img2 = np.array(Image.new('RGB', (size, size), (255, 255, 255)))

        if image_filter == 'aligned' and aligned_image.exists():
            # print(f'\n{image_path}:')
            btn = remove_button(str(aligned_image))
            display(widgets.HBox([ widgets.Label(value=f'\n{i}: {image_path}:'), btn], layout=widgets.Layout(margin='30px 0 0 0')))
            if landmarks:
                p = img1.shape[0] - img2.shape[0]
                img2 = np.pad(img2, pad_width=((0, p), (0, 0), (0, 0)), mode='constant', constant_values=255)
            img = Image.fromarray(np.concatenate([img1, img2], axis=1))
            display(img)

        if image_filter == 'missed' and not aligned_image.exists():
            print(f'\n{i}: {image_path}:')
            display(Image.fromarray(img1))
