import argparse
import torch


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument(
            "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0. use -1 for CPU"
        )
        self.parser.add_argument("--name", type=str, default="Baseline")
        self.parser.add_argument("--dataroot", type=str, default="/raid/jiangvicky/Dataset/SentinelData")
        self.parser.add_argument("--dataset", type=str, default="train_val")
        self.parser.add_argument(
            "--stat-dir", type=str, default="/raid/jiangvicky/Model/MVEO/data/statistics"
        )
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoints",
            help="models are saved here",
        )

        self.parser.add_argument(
            "--result_dir", type=str, default="./results", help="results are saved here"
        )
        self.parser.add_argument(
            "--vis_path", type=str, default="vis", help="results are saved here"
        )
        self.parser.add_argument("--load_pretrain", action="store_true")

        self.parser.add_argument("--phase", type=str, default="train")
        self.parser.add_argument("--input_size", type=int, default=128)
        self.parser.add_argument("--backbone", type=str, default="mobilenetv2")
        self.parser.add_argument("--fpn", type=str, default="fpn")
        self.parser.add_argument("--fpn_channels", type=int, default=96)
        self.parser.add_argument("--deform_groups", type=int, default=4)
        self.parser.add_argument("--gamma_mode", type=str, default="SE")
        self.parser.add_argument("--beta_mode", type=str, default="contextgatedconv")
        self.parser.add_argument("--num_heads", type=int, default=1)
        self.parser.add_argument("--num_points", type=int, default=8)
        self.parser.add_argument("--kernel_layers", type=int, default=1)
        self.parser.add_argument("--init_type", type=str, default="kaiming_normal")
        self.parser.add_argument("--alpha", type=float, default=0.25)
        self.parser.add_argument(
            "--gamma", type=int, default=4, help="gamma for Focal loss"
        )
        self.parser.add_argument("--dropout_rate", type=float, default=0.1)

        self.parser.add_argument("--batch_size", type=int, default=16)
        self.parser.add_argument("--num_epochs", type=int, default=200)
        self.parser.add_argument("--warmup_epochs", type=int, default=2)
        self.parser.add_argument(
            "--num_workers", type=int, default=4, help="#threads for loading data"
        )
        self.parser.add_argument("--lr", type=float, default=5e-4)
        self.parser.add_argument("--weight_decay", type=float, default=5e-4)
        self.parser.add_argument("--segmodel", type=str, default="base")
        self.parser.add_argument("--create_model", type=str, default="base")
        self.parser.add_argument(
            "--aug_mode",
            type=str,
            default="full",
            choices=["none", "noise_erase", "full"],
            help="full: geometry+noise+erase, noise_erase: only noise+erase, none: disable train-time augmentation",
        )
        self.parser.add_argument("--aug_p_noise", type=float, default=0.5)
        self.parser.add_argument("--aug_p_erase", type=float, default=0.5)
        self.parser.add_argument("--seed", type=int, default=77777)

        self.parser.add_argument(
            '--l1_weight',
            type=float,
            default=1,
            help='L1 regularization weight on classifier heads'
        )

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        return self.opt
