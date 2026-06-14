import os
import shutil
import numpy as np
import torch
from PIL import Image
from option import Options
from data.dataset import DataLoader
from model import create_model

from tqdm import tqdm

if __name__ == "__main__":
    opt = Options().parse()
    opt.phase = "test"
    opt.dataset = "test"
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    tbar = tqdm(test_data, ncols=80)
    total_iters = test_size

    model.eval()

    result_path = os.path.join(opt.checkpoint_dir, opt.name, opt.result_dir)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with torch.inference_mode():
        for i, _data in enumerate(tbar):
            val_pred = model.inference(_data["img1"].cuda(), _data["img2"].cuda())
            val_pred = torch.argmax(val_pred.detach(), dim=1)
            val_pred = val_pred.cpu().numpy()

            for i in range(val_pred.shape[0]):
                Image.fromarray(val_pred[i].astype(np.uint8)).save(
                    os.path.join(
                        result_path,
                        _data["fname"][i].replace(".tif", ".png"),
                    )
                )
    shutil.make_archive(result_path, "zip", result_path)
