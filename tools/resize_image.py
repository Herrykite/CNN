import sys

sys.path.insert(0, '../../')
import os
from ConvNet.config.defaults import get_cfg_defaults
from PIL import Image
from torchvision import transforms
from ConvNet.tools.preprocess_data import mkdir

if __name__ == '__main__':
    print('The photo data is being resized...')
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg.MODEL.CONFIG + os.listdir(cfg.MODEL.CONFIG)[-1])
    transform = transforms.Compose([
        transforms.Resize(cfg.DATASETS.TRANSFORM_RESIZE),
    ])

    image_path = cfg.DATASETS.IMAGES_PATH
    save_image_path = cfg.INPUT.SAVE_RESIZE_IMAGES
    images = os.listdir(image_path)
    mkdir(save_image_path)
    for i in range(len(images)):
        figure = Image.open(image_path + images[i]).convert('L')
        new_figure = transform(figure)
        new_figure.save(save_image_path + images[i])
