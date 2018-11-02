import argparse
import json
import os
import warnings

class GatherOptions():
    def __init__(self):
        parser = argparse.ArgumentParser(description="train or test CAGAN-v2")
        parser.add_argument("--mode", default="train", choices=["train", "test"],
                            help="train or test the model" )
        parser.add_argument("--data_root", default="data", help="path to images")
        parser.add_argument("--batchsize", type=int, default=8, help="input batch size")
        parser.add_argument("--num_workers", type=int, default=2, help="threads for loading data")
        parser.add_argument("--epoch", type=int, default=30, help="num of eopch for training")
        parser.add_argument("--save_dir", default="logs/origin", help="path for saving model weight and images")
        parser.add_argument("--model_dir", help="path to load model for test(the largest step or use --step to specify)")
        parser.add_argument("--resume", action="store_true", help="resume training, have to use --step to specify the initial step")
        parser.add_argument("--step", type=int, default=0, help="choose which step to load or resume")
        
        parser.add_argument('--save_model_freq', type=int, default=3000, help="frequency of saving model weight")
        parser.add_argument('--save_image_freq', type=int, default=500, help="frequency of saving image for visualization")
        parser.add_argument('--lr', type=float, default=0.0002, help="initial learning rate for adam")
        parser.add_argument('--lr_policy', type=str, default='lambda', help="learning rate policy: lambda|step|cosine")
        parser.add_argument('--niter', type=int, default=6000, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=-1, help='# of iter to linearly decay learning rate to zero, default at the end of training')
        parser.add_argument('--lr_decay_iters', type=int, default=7, help="multiply by a gamma every lr_decay_iters iterations")
        parser.add_argument('--up_type', type=str, default='Tconv', help='the type of upscale: Tconv|bilinear|nearest|ps (pixel shuffle)')
        parser.add_argument('--no_mixup', action='store_true', help="mixup real and fake before inputing netD")
        # parser.add_argument('--no_cycle', action='store_true', help='use cycleGAN in training')
        parser.add_argument('--use_lsgan', action='store_true', help='use least square GAN')
        parser.add_argument('--gamma_i', type=float, default=0.1, help="weight of id loss")
        parser.add_argument('--ngf', type=int, default=64, help="# of gen filters in first conv layer")
        parser.add_argument('--nc_G_inp', type=int, default=9, help="# of gen input channel ")

        self.parser = parser

    def parse(self, argv=None):
        if argv == None:
            opt = self.parser.parse_args(argv) # for running in jupyter notebook    
        else:
            opt = self.parser.parse_args()
        self.opt = opt
        self.config_path = os.path.join(opt.save_dir, 'opt.json')

        if opt.resume and (opt.step == 0):
            raise Exception("Please use --step to specify the initial step!")
        if opt.mode == "test":
            self.compare_config_test(['up_type', 'ngf', 'nc_G_inf'])
        elif opt.resume:
            self.compare_config_train(['mode', 'save_dir', 'model_dir', 'resume',
                                       'step', 'save_model_freq', 'save_image_freq'])
        else:
            with open(self.config_path, 'w') as f:
                json.dump(self.opt.__dict__, f)

        return opt

    def compare_config_train(self, except_names):
        try:
            with open(self.config_path, 'r') as f:
                train_opt_dict = json.load(f)
        except FileNotFoundError:
            warnings.warn("Not Found opt.json in %s"%self.opt.save_dir)
        else:
            for key, data in train_opt_dict.items():
                if key not in except_names:
                    if self.opt.__dict__[key] != data:
                        warnings.warn('model config is conflic, %s should be \'%s\'!'%(key, data))


    def compare_config_test(self, config_names):
        try:
            with open(self.config_path, 'r') as f:
                train_opt_dict = json.load(f)
        except FileNotFoundError:
            warnings.warn("Not Found opt.json in %s"%self.opt.save_dir)
        else:
            for key, data in train_opt_dict.items():
                if key in config_names:
                    if self.opt.__dict__[key] != data:
                        raise Exception('model config is conflic, %s should be \'%s\'!'%(key, data))