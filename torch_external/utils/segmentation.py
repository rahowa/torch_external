
import numpy as np 
import pandas as pd 
from typing import List, Dict
from copy import deepcopy
from collections import deque

DataFrame = pd.DataFrame
Image = np.ndarray


__all__ = ["mask2rle", "rle2mask", "blend_rle"]


def mask2rle(img: Image) -> str:
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle: str, shape: List[int]) -> Image:
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if mask_rle != mask_rle:
        return np.zeros_like(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def blend_rle(submits: List[DataFrame], image_shapes: Dict[str, List[int]]) -> DataFrame:
    tmp_result = deque(maxlen=len(submits))
    result = deepcopy(submits[0])
    tmp_rles = deque(maxlen=submits[0].shape[0])
    for idx in range(submits[0].shape[0]):
        for submit in submits:
            image_shape = image_shapes[result.iloc[idx]['ImageId']]
            decoded_mask = rle2mask(submit.iloc[idx]['EncodedPixels'], 
                                     (image_shape[1], image_shape[0]))

            if decoded_mask.shape[0] >= 384:
                tmp_result.append(decoded_mask)

        mean_image = np.where(np.array(tmp_result).mean(0) > 0.5, 1, 0)

        if mean_image.mean() == 0:
            mean_image = np.where(np.array(tmp_result).mean(0) > 0.1, 1, 0)
        tmp_rles.append(mask2rle(mean_image.T))
        tmp_result.clear()
    result['EncodedPixels'] = tmp_rles
    return result

