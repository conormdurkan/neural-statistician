import argparse
import glob
import os

from scipy.misc import imsave
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', require=True, type=str, default=None)
args = parser.parse_args()
assert (args.data_dir is not None) and (os.path.isdir(args.data_dir))

filenames = glob.glob(os.path.join(args.data_dir, '/original/**/*.jpg'),
                      recursive=True)

for filename in tqdm(filenames):
    image = imread(filename)
    x, y, _ = image.shape
    x = int(x / 2)
    y = int(y / 2)
    image = image[(x - 64):(x + 64), (y - 64):(y + 64), :]
    image = resize(image, output_shape=(64, 64), mode='reflect')
    file = filename.replace('original', '64')
    try:
        imsave(file, image)
    except:
        directory = '/'.join(file.split('/')[:-1])
        os.makedirs(directory, exist_ok=True)
        imsave(file, image)