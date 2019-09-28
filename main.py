from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import tensorflow as tf


class DigitImageMaker:
    def __init__(self):
        # TODO: for "font" mode, generated font & images can be cached to reuse
        # Setting ####
        self.font = "FE-FONT"
        self.min_digit_width = 10
        self.y_perturb = 10

        # Member variable ####
        self.digit_to_mnist = self._load_mnist()  # str -> [numpy.array,]

    def create_digit_sequence(self, number, image_width,
                              min_spacing, max_spacing,
                              generation_mode="mnist-no-resize"):
        """A function that creates an image representing the given number, with
        random sampling between the digits.

        Generation of each digit depends on `generation_mode`:
            "font": digit is generated from font, width depends on max_spacing
            "mnist-no-resize": digit is sampled from MNIST, width is constant

        This function does not do padding. TODO: do random padding if necessary

        Parameters
        ----------
        number: str
            A string representing the number, e.g. "14543"
        image_width: int
            The image width (in pixel).
        min_spacing: int
            The minimum spacing between digits (in pixel).
        max_spacing: int
            The maximum spacing between digits (in pixel).
        generation_mode: str
            The mode of generating digit.

        Returns
        ----------
        NumPy array of the final image. (np.array)
        """
        # no padding, so padding should be done outside this function
        n = len(number)

        if generation_mode == "font":
            max_digit_width = (image_width - (n - 1) * max_spacing) // n
            if max_digit_width < self.min_digit_width:
                raise RuntimeError("len(number) * max_spacing > image_width ")

            digit_images = [[self._digit_to_np_image(str(i),
                     "fonts/FE-FONT.TTF", max_digit_width)] for i in range(10)]
            image_height = max(
                     [digit_image[0].shape[0] for digit_image in digit_images])
        elif generation_mode == "mnist-no-resize":
            # load mnist dataset
            digit_images = self.digit_to_mnist
            image_height = 28
        image_height += self.y_perturb

        final_image = np.zeros((image_height, image_width, 3))
        y, x = 0, 0
        for digit in number:
            y = int(random.uniform(0, self.y_perturb))
            digit_image = random.choice(digit_images[int(digit)])
            digit_height, digit_width, _ = digit_image.shape
            final_image[y:y + digit_height, x:x + digit_width, :] = digit_image
            spacing = int(random.uniform(min_spacing, max_spacing))
            x += (digit_width + spacing)

            # TODO: perturb y slightly in range (Method 2 to prevent overfitting)
        # TODO: not necessary but maybe center the digit seq since final
        # TODO: image can have a lot of whitespace on the right
        return final_image

    def _load_mnist(self):
        """Load mnist from tensorflow for self.digit_to_mnist dictionary.
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        n = y.shape[0]
        digit_to_mnist = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [],
                          7: [], 8: [], 9: []}

        for i in range(n):
            # duplicate to 3 channels
            new_xi = np.stack((x[i], x[i], x[i]), axis=-1)
            digit_to_mnist[y[i]].append(new_xi)

        return digit_to_mnist

    def _digit_to_np_image(self, str_digit, font_path, output_width):
        """str -> np.array (uint8) image representing the integer using font

        Parameters
        ----------
        str_digit: str
            0~9 digit in string format.
        font_path: str
            Path of font to use.
        output_width: int
            The width (in pixels) of the output digit image.

        Returns
        ----------
        Numpy array of the digit image. (np.array, dtype=uint8)
        """
        # *8 so font point is large enough to be scaled down later
        font_pt = output_width * 8
        font = ImageFont.truetype(font_path, font_pt)
        width, height = font.getsize(str_digit)
        im = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), str_digit, fill=(255, 255, 255), font=font)
        im = im.resize((output_width, height * output_width // width))
        return np.array(im)


def main():
    s = "929394"
    mode = "font"
    dim = DigitImageMaker()
    res = dim.create_digit_sequence(s, 50*6+5*20, 1, 20, mode)
    im = Image.fromarray(res.astype(np.uint8))

    im.save(s + mode + ".png")


main()
