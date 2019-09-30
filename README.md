# DigitImageMaker

- Skill test for Mercari Software Engineer, Machine Learning internship
- Basic functionalities with some other extensions are implemented. I have other ideas to extend/optimize this project which is listed in a later section.

## Requirements 

- Python 3 
- imutils
- matplotlib
- numpy
- pandas
- pillow
- tensorflow
- tqdm



## How to Use

#### The basic setting

To test `create_digit_sequence()`directly,  run`python main.py --number=NUMBER --image_width=IMAGE_WIDTH --min_spacing=MIN_SPACING --max_spacing=MAX_SPACING`. An image with the name `0.png` will be created inside the folder `images`. `id_label.csv` contains the label (`number`) of the image. 

#### Implemented extensions

I have added some functionalities with OCR data augmentation in mind. More specifically, I made following extensions that are accessible by command line arguments:

- `--gaussian_noise` Add a Gaussian noise to the image
- `--random_rotation=ANGLE` Rotate the image by angle uniformly sampled from `-ANGLE ~ +ANGLE`. Rotation can change the height of the final image but the width is as specified by `--image_width.`
- `--random_color` Add colors to the grayscale image with randomly chosen color maps from `matplotlib`.
- `--y_perturb=R` Add random perturbations in the range `0 ~ +R` on y-axis for each digit. This is to make up for the fact that `--min-spacing` and `--max_spacing` only deal with x-axis perturbations. 
- `--generation_mode=MODE` Allow to generate texts based on MNIST or fonts. MNIST is useful for hand written digit, but if we want OCR to recognize printed text too, some text image dataset would be nice. Set this argument to `font` for generation based on fonts, `mnist_no_resize` for MNIST generation based on MNIST. (Text size of each digit in`font` mode is automatically calculated from `IMAGE_WIDTH` and `MAX_SPACING`. In contrast, size of each digit in `mnist_no_resize` is always 28x28 )

To see these options in action, try these demo by following commands:

- `python main.py --demo_vanilla --min_spacing=MIN_SPACING --max_spacing=MAX_SPACING`
  - This creates 100 random images with above options disabled.
- `python main.py --demo_hand_written --min_spacing=MIN_SPACING --max_spacing=MAX_SPACING`
  - This creates 100 MNIST images with above options enabled.
- `python main.py --demo_font --min_spacing=MIN_SPACING --max_spacing=MAX_SPACING`
  - This creates 100 images from a font with above options enabled.

#### Other features

Here are other features accessible via CLI arguments:

- `--dataset_name=NAME ` This specifies the name of the dataset being created. By default this is `"images"`. Pictures and csv with label information will be created in the folder specified by this arguments.
- `--num_image=NUM_IMAGE`This specifies how many images to be created in this run. By default this is `1`.
- `--number=NUMBER` This is the same as the `number` argument for `create_digit_sequence()`. If this value is set, all images will be based on this number. If this value is **not** **set**, **random number** from 0~9999999999 is used for each image.
  ​



## Other Possible Extensions Not Implemented 

The following are examples of other possible extensions / data augmentation techniques I thought of but could not finish due to time constraints.

- **Add natural backgrounds in the back or irrelevant objects around the text** 
  - This is useful if OCR needs to identify texts with complicated background or surrounding, which is quite common.
- **Do perspective transform**
  - A good OCR should be able to deal with reasonable x-axis, y-axis, z-axis transformations, like watching from 45 degrees angle.
- **Adjust lighting condition**
  - Human eyes can identify texts regardless of brightness.
- **Motion Blur**
  - It would be ideal if OCR can deal with slight motion blur, which is common when humans are taking pictures.
- **Further optimize performance**
  - The code is written with correctness and cleanliness in mind. The performance can be further optimized for large-scale machine learning scenario.
- **Better spacing sampling strategy**
  - Currently the spacing is uniformly sampled from `MIN_SPACING ~ MAX_SPACING`. To prevent errors, I forced `IMAGE_WIDTH` to be larger than `MAX_SPACING * (n - 1) + digit_width * n` where `n` is the length of `NUMBER`. This causes an issue that if gap between `MIN_SPACING` and `MAX_SPACING` is large, there can be a lot of empty space on the right. Maybe a spacing sampling strategy closer to human writing behavior (instead of the simple uniform sampling) would be ideal (maybe Gaussian distribution).
- **Generation mode that supports custom images**
  - If OCR system wants to recognize a specific type of texts instead of MNIST or fonts, then there is a necessity to let this script support custom images.
- **Random padding** 
  - Do random paddings on the surrounding for more robust OCR.