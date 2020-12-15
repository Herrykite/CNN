import os
from ConvNet.config.defaults import get_cfg_defaults
from PIL import Image
from torchvision import transforms
from ConvNet.tools.preprocess_data import mkdir

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cfg = get_cfg_defaults()
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
