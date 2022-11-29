# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Logging utils
"""

import os
import warnings

import pkg_resources as pkg
import torch
from torch.utils.tensorboard import SummaryWriter


from utils.general import colorstr, cv2, emojis
from utils.plots import plot_images, plot_results
from utils.torch_utils import de_parallel

LOGGERS = ('csv', 'tb')  # text-file, TensorBoard, Weights & Biases
RANK = int(os.getenv('RANK', -1))

class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP_0.5', 'best/mAP_0.5:0.95']
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # TensorBoard
        s = self.save_dir
        # if 'tb' in self.include and not self.opt.evolve:
        #     prefix = colorstr('TensorBoard: ')
        #     self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
        #     self.tb = SummaryWriter(str(s))

        self.wandb = None
    def on_train_start(self):
        # Callback runs on train start
        pass

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        paths = self.save_dir.glob('*labels*.jpg')  # training labels

    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots):
        # Callback runs on train batch end
        if plots:
            if ni == 0:
                if self.tb and not self.opt.sync_bn:  # --sync known issue https://github.com/ultralytics/yolov5/issues/3754
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')  # suppress jit trace warning
                        self.tb.add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                plot_images(imgs, targets, paths, f)


    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        pass

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        pass

    def on_val_end(self):
        # Callback runs on val end
        pass

    def on_fit_epoch_end(self, vals, epoch):
        # Callback runs at the end of each fit (train+val) epoch

        keys = [
            'metrics/precision@.5',
            'metrics/recall@.5',
            'metrics/f1@.5',
            'metrics/precision@.5:.95',
            'metrics/recall@.5:.95',
            'metrics/f1@.5:.95',
            'x/lr0',
            'x/lr1',
            'x/lr2'] 

        x = dict(zip(keys, vals))
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        pass

    def on_train_end(self, last, best, plots, epoch, results):
        # Callback runs on training end
        if plots:
            plot_results(file=self.save_dir / 'results.csv')  # save results.png
        files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter
        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        if self.tb:
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')

    def on_params_update(self, params):
        pass


