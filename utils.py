import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import pickle

def print_images(source_folder='cartoon-images', target_folder='cartoon-images-aligned', image_filter='aligned', landmarks=False, size=256):

    image_types = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    source_images = [str(p) for p in Path(source_folder).glob('**/*.*') if p.suffix in image_types]

    for image_path in source_images:
        if landmarks:
            image = Image.open(image_path)
            landmarks_file = Path(target_folder + '/' + str(Path(image_path).stem) + '_00.pkl')
            if landmarks_file.exists():
                infile = open(landmarks_file, 'rb')
                rect, landmarks = pickle.load(infile)
                infile.close()
                image = np.array(image)
                cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
                for landmark in landmarks:
                    cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)
                image = Image.fromarray(image)
            img1 = image
        else:
            img1 = Image.open(image_path).resize((size, size))

        aligned_image = Path(target_folder + '/' + str(Path(image_path).stem) + '_00.png')

        if aligned_image.exists():
            img2 = Image.open(str(aligned_image)).resize((size, size))
        else:
            img2 = Image.new('RGB', (size, size), (255, 255, 255))

        if image_filter == 'aligned' and aligned_image.exists():
            if landmarks:
                p = np.array(img1).shape[0] - np.array(img2).shape[0]
                img2 = np.pad(np.array(img2), pad_width=((0, p), (0, 0), (0, 0)), mode='constant', constant_values=255)
            img = Image.fromarray(np.concatenate([np.array(img1), np.array(img2)], axis=1))
            print(f'\n{image_path}:')
            display(img)

        if image_filter == 'missed' and not aligned_image.exists():
            print(f'\n{image_path}:')
            display(img1)

