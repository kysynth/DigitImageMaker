from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

MIN_DIGIT_WIDTH = 10

def create_digit_sequence(number, image_width, min_spacing, max_spacing):
    # no padding, so padding should be done outside this function
    n = len(number)
    # maximum pixel width allowed for each digit
    # TODO: randomize each font width slightly (Method 1 to prevent overfitting)
    max_digit_width = (image_width - (n - 1) * max_spacing) // n
    if max_digit_width < MIN_DIGIT_WIDTH:
        raise RuntimeError("len(number) * max_spacing > image_width ")

    # TODO: refactor below num_to_np_images out
    digit_images = [digit_to_np_image(str(i), "fonts/FE-FONT.TTF",
                                     max_digit_width) for i in range(10)]
    image_height = max([digit_image.shape[0] for digit_image in digit_images])

    final_image = np.ones((image_height, image_width, 3)) * 255
    y, x = 0, 0
    for digit in number:
        digit_image = digit_images[int(digit)]
        digit_height, digit_width, _ = digit_image.shape
        final_image[y:y + digit_height, x:x + digit_width, :] = digit_image
        spacing = int(random.uniform(min_spacing, max_spacing))
        x += (digit_width + spacing)
        # TODO: perturb y slightly in range (Method 2 to prevent overfitting)
    # TODO: not necessary but maybe center the digit seq since final
    # TODO: image can have a lot of whitespace on the right
    return final_image


def digit_to_np_image(str_digit, font_path, output_width):
    """str -> np.array (uint8) image representing the integer using font

    Parameters
    ----------
    str_digit: str
        0~9 number in string format
    font_path: str
        path of font to use

    Returns
    ----------
    xxxxx
    """
    # *8 so font point is large enough to be scaled down later
    font_pt = output_width * 8
    font = ImageFont.truetype(font_path, font_pt)
    width, height = font.getsize(str_digit)
    im = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), str_digit, fill=(0, 0, 0), font=font)
    im = im.resize((output_width, height * output_width // width))
    im.save(str(output_width) + '.png')
    return np.array(im)


def main():
    s = "196027"
    res = create_digit_sequence(s, 100*6+5*500, 1, 500)
    im = Image.fromarray(res.astype(np.uint8))

    im.save(s + ".png")


main()
