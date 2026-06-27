import numpy as np
from PIL import Image

data_dir = "/home/intlab/d_drive/dataset/SYSU_MM01/"

def make_fixed_288x144(src_name, dst_name):
    imgs = np.load(data_dir + src_name, allow_pickle=True)
    out = []

    for i, img in enumerate(imgs):
        img = img.astype(np.uint8)
        img = Image.fromarray(img).resize((144, 288), Image.BICUBIC)  # PIL: (W, H)
        out.append(np.array(img, dtype=np.uint8))

        if i % 1000 == 0:
            print(i, "/", len(imgs))

    out = np.stack(out, axis=0)
    print(dst_name, out.shape, out.dtype)
    np.save(data_dir + dst_name, out)

make_fixed_288x144(
    "train_rgb_resized_img.npy",
    "train_rgb_resized_img_288.npy"
)

make_fixed_288x144(
    "train_ir_resized_img.npy",
    "train_ir_resized_img_288.npy"
)