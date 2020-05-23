from PIL import Image
from os import *

new_idx = 1

in_file_dir = r"./rawImg/horse_painting/"  # 需要改变的文件目录
out_file_dir = "datasets/trainB/paintHorse_"  # 改变后的文件目录

for filename in listdir(in_file_dir):
    try:
        im = Image.open(in_file_dir + filename)
        nx, ny = im.size
        im2 = im.resize((256,256), Image.BICUBIC)
        im2.save(out_file_dir + str(new_idx) + ".jpg")
        new_idx += 1
    except Exception as e:
        print(e)
        pass


