from maker import DigitImageMaker
from PIL import Image
from tqdm import tqdm
import argparse
import os
import pandas as pd
import random


def main():
    """The main part of the program that write image files according to CLI
    arguments.
    """
    args = get_args()
    if args.demo_vanilla:
        args.generation_mode = 'mnist_no_resize'
        args.num_image = 100
        args.gaussian_noise = False
        args.random_rotation = 0
        args.random_color = False
        args.y_perturb = 0
    elif args.demo_hand_written:
        args.generation_mode = 'mnist_no_resize'
        args.num_image = 100
        args.gaussian_noise = True
        args.random_rotation = 10
        args.random_color = True
        args.y_perturb = 5
    elif args.demo_font:
        args.generation_mode = 'font'
        args.num_image = 100
        args.gaussian_noise = True
        args.random_rotation = 10
        args.random_color = True
        args.y_perturb = 5

    follow_cli_args(args)


def get_args():
    """Add arguments and return parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--demo_vanilla', action='store_true', help='Demo of basic generations of hand written seq.')
    parser.add_argument('--demo_hand_written', action='store_true', help='Demo of more randomized hand written texts.')
    parser.add_argument('--demo_font', action='store_true', help='Demo of generating randomized font texts.')

    parser.add_argument('--number', type=str, help='Digit sequence to make images of. Leave this empty for random seq.')
    parser.add_argument('--image_width', type=int, help='Image width of the output images.')
    parser.add_argument('--min_spacing', required=True, type=int, help='Minimum spacing between digits.')
    parser.add_argument('--max_spacing', required=True, type=int, help='Maximum spacing between digits.')
    parser.add_argument('--generation_mode', type=str, default='mnist_no_resize', help='Gen images via MNIST or fonts.')

    parser.add_argument('--dataset_name', type=str, default='images', help='Images will be in a folder with this name.')
    parser.add_argument('--num_image', type=int, default=1, help='Number of images to create.')
    parser.add_argument('--gaussian_noise', action='store_true', help='Add gaussian noise to images.')
    parser.add_argument('--min_digit_width',type=int, default=10,help='Only for `font` mode,minimum digit pixel width.')
    parser.add_argument('--random_rotation', type=int, default=0, help='Specify max angles to rotate digit sequences.')
    parser.add_argument('--random_color',action='store_true',help='Assign random color to images; otherwise grayscale.')
    parser.add_argument('--y_perturb', type=int, default=0, help='Y coordinate randomization in pixels.')

    # TODO: can catch arguments error like non-digits in --number

    return parser.parse_args()


def demo_hand_written():
    """Demo usage of creating hand written digits dataset.
    """


def demo_font():
    """Demo usage of creating digits with font.
    """


def follow_cli_args(args):
    """Generate images as specified by CLI arguments.
    """
    print("Following command line arguments.")
    print("Initializing the image maker...")
    dim = DigitImageMaker(args)

    id_label_dict = {'id': [], 'label': []}
    os.mkdir(args.dataset_name)

    for id_ in tqdm(range(args.num_image)):
        if args.number is None:
            number = str(random.randint(0, 9999999999))
        else:
            number = args.number
        if args.image_width is None:
            image_width = len(number) * (args.max_spacing + 28)
        else:
            image_width = args.image_width

        image = dim.create_digit_sequence(number, image_width, args.min_spacing,
                                          args.max_spacing,args.generation_mode)
        im = Image.fromarray(image)
        im.save(os.path.join(args.dataset_name, str(id_) + ".png"))

        id_label_dict['id'].append(id_)
        id_label_dict['label'].append(number)

    df = pd.DataFrame(id_label_dict)
    df.to_csv(os.path.join(args.dataset_name, "id_label.csv"), index=False)


main()