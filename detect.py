import json
from pathlib import Path
from typing import Dict

import click
import cv2 as cv
import numpy as np
from tqdm import tqdm


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    img = cv.imread(img_path, cv.IMREAD_COLOR)

    ######################################################################## Image resize
    h, w = img.shape[:2]
    x = h / 1000.0
    img = cv.resize(img, (int(w / x), int(h / x)))

    channels = cv.split(img)

    ######################################################################## Shadows removal
    output_channels = []
    kernel_size = 31
    blur_size = 31

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for channel in channels:
        img_dil = cv.dilate(channel, kernel, iterations=1)
        img_bg = cv.medianBlur(img_dil, blur_size)
        img_ad = 255 - cv.absdiff(channel, img_bg)
        norm_img = cv.normalize(img_ad, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        output_channels.append(norm_img)

    img_ns = cv.merge(output_channels)

    ######################################################################## Background removal
    gray = cv.cvtColor(img_ns, cv.COLOR_RGB2GRAY)

    thresholdlvl = 142
    _, th = cv.threshold(gray, thresholdlvl, 255, cv.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    closing_iter = 2
    opening_iter = 2
    th = cv.morphologyEx(th, cv.MORPH_CLOSE, kernel, iterations=closing_iter)
    th = cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations=opening_iter)
    nobg = cv.bitwise_and(img, img, mask=th)

    nobg = cv.GaussianBlur(nobg, (5, 5), 0)

    hsv_img = cv.cvtColor(nobg, cv.COLOR_RGB2HSV)

    ######################################################################## Color masks
    yellow_range = ((93, 220, 80), (101, 255, 255))
    yellow_mask = cv.inRange(hsv_img, yellow_range[0], yellow_range[1])
    yellow_only = cv.bitwise_and(img, nobg, mask=yellow_mask)

    green_range = ((69, 210, 20), (90, 255, 170))
    green_mask = cv.inRange(hsv_img, green_range[0], green_range[1])
    green_only = cv.bitwise_and(img, nobg, mask=green_mask)

    red_range = ((117, 175, 45), (125, 255, 245))
    red_mask = cv.inRange(hsv_img, red_range[0], red_range[1])
    red_only = cv.bitwise_and(img, nobg, mask=red_mask)

    violet_range = ((125, 100, 20), (158, 255, 210))
    violet_mask = cv.inRange(hsv_img, violet_range[0], violet_range[1])
    violet_only = cv.bitwise_and(img, nobg, mask=violet_mask)

    ######################################################################## Thresholding
    _, yellow_th = cv.threshold(cv.cvtColor(yellow_only, cv.COLOR_RGB2GRAY), 0, 255, cv.THRESH_BINARY)
    yellow_th = cv.morphologyEx(yellow_th, cv.MORPH_OPEN, kernel, iterations=2)
    yellow_th = cv.morphologyEx(yellow_th, cv.MORPH_CLOSE, kernel, iterations=1)

    _, green_th = cv.threshold(cv.cvtColor(green_only, cv.COLOR_RGB2GRAY), 0, 255, cv.THRESH_BINARY)
    green_th = cv.morphologyEx(green_th, cv.MORPH_OPEN, kernel, iterations=2)
    green_th = cv.morphologyEx(green_th, cv.MORPH_CLOSE, kernel, iterations=1)

    _, red_th = cv.threshold(cv.cvtColor(red_only, cv.COLOR_RGB2GRAY), 0, 255, cv.THRESH_BINARY)
    red_th = cv.morphologyEx(red_th, cv.MORPH_OPEN, kernel, iterations=2)
    red_th = cv.morphologyEx(red_th, cv.MORPH_CLOSE, kernel, iterations=1)

    _, violet_th = cv.threshold(cv.cvtColor(violet_only, cv.COLOR_RGB2GRAY), 0, 255, cv.THRESH_BINARY)
    violet_th = cv.morphologyEx(violet_th, cv.MORPH_OPEN, kernel, iterations=2)
    violet_th = cv.morphologyEx(violet_th, cv.MORPH_CLOSE, kernel, iterations=1)

    ######################################################################## Contours
    yellow_con, yellow_h = cv.findContours(yellow_th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    green_con, green_h = cv.findContours(green_th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    red_con, red_h = cv.findContours(red_th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    violet_con, violet_h = cv.findContours(violet_th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    red = len(red_con)
    yellow = len(yellow_con)
    green = len(green_con)
    purple = len(violet_con)

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
