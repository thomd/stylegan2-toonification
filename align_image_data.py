import os
import pickle
import argparse
from pathlib import Path
from stylegan2.ffhq_dataset.face_alignment import image_align
from tqdm import tqdm
import numpy as np
import dlib
import gdown


def list_images(images_path):
    image_types = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_paths = [str(p) for p in Path(images_path).glob('**/*.*') if p.suffix in image_types]
    return image_paths


def align_images(args):
    source_folder = list_images(args['images_source'])
    target_folder = Path(args['images_target'])
    target_folder.mkdir(exist_ok=True)

    if args['model'] == 'hog':
        face_detector = dlib.get_frontal_face_detector()
    else:
        mmod_model = Path('mmod_human_face_detector.dat')
        if not mmod_model.exists():
            gdown.download(id='1oGNn74w9zU77uEVgzPrLxDG6X8aPzvba')
        face_detector = dlib.cnn_face_detection_model_v1(str(mmod_model))

    # face landmarks model
    landmarks_model = Path('shape_predictor_68_face_landmarks.dat')
    if not landmarks_model.exists():
        gdown.download(id='1HChdZjXEIqgqilqU2ar_mMOk-JflK5ah')
    face_predictor = dlib.shape_predictor(str(landmarks_model))

    print(f'\nAligning {len(source_folder)} images using the \'{args["model"]}\' model')
    for source_image in tqdm(source_folder):
        image = dlib.load_rgb_image(source_image)
        faces = face_detector(image, 1)
        for i, face in enumerate(faces):
            rect = face if args['model'] == 'hog' else face.rect
            face_landmarks = [(item.x, item.y) for item in face_predictor(image, rect).parts()]
            target_image = str(target_folder / f'{Path(source_image).stem}_{i:02d}.png')
            image_align(source_image, target_image, face_landmarks, output_size=args['output_size'])
            if args['metadata']:
                target_metadata = str(target_folder / f'{Path(source_image).stem}_{i:02d}.pkl')
                outfile = open(target_metadata, 'wb')
                pickle.dump((rect, face_landmarks), outfile)
                outfile.close()


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description='Pre-process image data', formatter_class=CustomFormatter)
    parser.add_argument('--images-source', default='cartoon-images', type=Path, metavar='PATH', help='path to cartoon images source')
    parser.add_argument('--images-target', default='cartoon-images-aligned', type=Path, metavar='PATH', help='path to aligned cartoon images')
    parser.add_argument('--output-size', default=1024, type=int, metavar='SIZE', help='target image size')
    parser.add_argument('--model', default='mmod', choices=['hog', 'mmod'], help='face detector model')
    parser.add_argument('--metadata', action='store_true', default=True, help='store metadata (faces, landmarks)')
    args = vars(parser.parse_args())

    align_images(args)

