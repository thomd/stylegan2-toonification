import config
import argparse
import pathlib

def process_data(args):
    pass

if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description='Pre-process image data', formatter_class=CustomFormatter)
    parser.add_argument('--images-path', type=pathlib.Path, default=config.IMAGES_PATH, metavar='PATH', help='path to image data source')
    parser.add_argument('--dataset-path', type=pathlib.Path, default=config.DATASET_PATH, metavar='PATH', help='path to dataset for transfer learning')
    args = vars(parser.parse_args())

    process_data(args)



