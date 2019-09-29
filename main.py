from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm
import numpy as np
import random
import tensorflow as tf


class DigitImageMaker:
    def __init__(self):
        # TODO: for "font" mode, generated font & images can be cached to reuse
        # Setting ####
        self.font = "FE-FONT"
        self.gaussian_noise = True
        self.min_digit_width = 10
        self.random_color = True
        self.y_perturb = 5


        # Member variable ####
        self.digit_to_mnist = self._load_mnist()  # str -> [numpy.array,]
        # random color
        self.color_maps = ['Greys', 'Purples', 'Blues', 'Greens', 'Reds',
          'PiYG', 'PRGn', 'BrBG', 'RdGy', 'RdBu', 'Pastel1', 'Pastel2',
          'Paired', 'Accent', 'flag', 'prism', 'ocean', 'terrain', 'gist_stern']
        # test

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
        if min_spacing > max_spacing:
            raise RuntimeError("min_width > max_width is invalid.")
        n = len(number)

        if generation_mode == "font":
            max_digit_width = (image_width - (n - 1) * max_spacing) // n
            if max_digit_width < self.min_digit_width:
                raise RuntimeError("image_width is too short for 'font' mode.")

            digit_images = [[self._digit_to_np_image(str(i),
                     "fonts/FE-FONT.TTF", max_digit_width)] for i in range(10)]
            image_height = max(
                     [digit_image[0].shape[0] for digit_image in digit_images])
        elif generation_mode == "mnist-no-resize":
            # load mnist dataset
            if image_width < (self.digit_to_mnist[0][0].shape[0]+max_spacing)*n:
                raise RuntimeError("image_width is too short for 'mnist-no-resize' mode.")
            digit_images = self.digit_to_mnist
            image_height = 28
        else:
            raise RuntimeError("No such generation_mode.")
        image_height += self.y_perturb

        final_image = np.zeros((image_height, image_width))
        y, x = 0, 0
        for digit in number:
            y = int(random.uniform(0, self.y_perturb))
            digit_image = random.choice(digit_images[int(digit)])
            digit_height, digit_width = digit_image.shape
            final_image[y:y + digit_height, x:x + digit_width] = digit_image
            spacing = int(random.uniform(min_spacing, max_spacing))
            x += (digit_width + spacing)

        # TODO: not necessary but maybe center the digit seq since final
        # TODO: image can have a lot of space on the right

        if self.random_color:
            random_color = random.choice(self.color_maps)
            cm = matplotlib.cm.get_cmap(random_color)
            final_image = cm(final_image) * 255
        if self.gaussian_noise:
            final_image = final_image.astype(np.float32)
            scale = 8 if self.random_color else 32
            final_image += np.random.normal(0, scale, final_image.shape)
            final_image = np.clip(final_image, 0, 255)

        return np.uint8(final_image)

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
            digit_to_mnist[y[i]].append(x[i])

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
        return np.squeeze(np.array(im)[:, :, 0])


def main():
        dim = DigitImageMaker()
        for i in range(20):
            print(i)
            s = "19970315"
            mode = "mnist-no-resize"
            res = dim.create_digit_sequence(s, 70*len(s), 1, 30, mode)
            im = Image.fromarray(res.astype(np.uint8))
            print("Saving...")
            im.save(s + mode + str(i) + ".png")


main()
