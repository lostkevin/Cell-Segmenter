import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

from predict import predict_generator, predict_generator_omni

from utils.general import LOGGER, save_as_csv
from compute_metric import eval_one_multithres


def evaluate(
    model: nn.Module,
    dataset: Dataset,
    *,
    timer: bool = False,
    num_workers = 4,
    evaluate_mask: bool = False,
    conf: float = 0.25,
    iou_thres: float = 0.2
):
    # Initialize/load model and set device
    model.eval()
    valloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    pred_gen = predict_generator(
        model, 
        valloader,
        conf=conf,
        iou_thres=iou_thres
    )
    results = {} # Type: Dict[str, Dict[str, float]], metrics (per-image) include: Precision Recall F1-score [time]
    for filename_noext, mask, gt, imgtype, *_ in pred_gen:
        r = eval_one_multithres(mask, gt.cpu().numpy())
        results[filename_noext] = {
            'precision_0.5': r[0, 0],
            'recall_0.5': r[0, 1],
            'f1_0.5': r[0, 2],
            'precision': r[:, 0].mean(),
            'recall': r[:, 1].mean(),
            'f1': r[:, 2].mean(),
            'imgtype': imgtype
        }
    return results

def evaluate_omni(
    model: nn.Module,
    dataset: Dataset,
    *,
    timer: bool = False,
    num_workers = 4,
    evaluate_mask: bool = False,
    flow_threshold=0.4,
    velocity=1
):
    # Initialize/load model and set device
    model.eval()
    valloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    pred_gen = predict_generator_omni(
        model, 
        valloader,
        flow_threshold=flow_threshold,
        velocity=velocity
    )
    results = {} # Type: Dict[str, Dict[str, float]], metrics (per-image) include: Precision Recall F1-score [time]
    for filename_noext, mask, gt in pred_gen:
        r = eval_one_multithres(mask, gt.cpu().numpy())
        results[filename_noext] = {
            'precision_0.5': r[0, 0],
            'recall_0.5': r[0, 1],
            'f1_0.5': r[0, 2],
            'precision': r[:, 0].mean(),
            'recall': r[:, 1].mean(),
            'f1': r[:, 2].mean(),
            # 'pa': (mask == gt).mean()
        }
    return results


def main(args):
    from models.experimental import attempt_load
    from utils.dataloaders import CellSegEvalDataset
    model: nn.Module = attempt_load(args.ckpt, fuse=False).eval().cuda()
    imageset = CellSegEvalDataset(args.input_path, args.overlap, True)
    if args.omni:
        results = evaluate_omni(model, 
                            imageset, 
                            evaluate_mask=args.mask, flow_threshold=args.flow, velocity=args.velo)
    else:
        results = evaluate(model, 
                    imageset, 
                    evaluate_mask=args.mask,
                    conf=args.conf,
                    iou_thres=args.iou_thres)
    save_as_csv(args.output_path, results)
    metrics = []
    for r in results.values():
        metrics.append(np.array([
            r['precision_0.5'],
            r['recall_0.5'],
            r['f1_0.5'],
            r['precision'],
            r['recall'],
            r['f1']
        ]))
    metrics = np.stack(metrics, axis=0).mean(axis=0)
    LOGGER.info('P@.5: {:.5f} R@.5: {:.5f} F1@.5: {:.5f}'.format(*metrics[:3]))
    LOGGER.info('P@.5:.95: {:.5f} R@.5:.95: {:.5f} F1@.5:.95: {:.5f}'.format(*metrics[-3:]))
    pass

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='./inputs', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument('-o', '--output_path', default='./outputs/results.csv', type=str, help='output filename')
    parser.add_argument('--ckpt', default=None, required=True)

    # Model parameters
    parser.add_argument('--input_size', default=640, type=int, help='segmentation classes')
    parser.add_argument('--overlap', default=0.125, type=float, help='overlap factor')
    parser.add_argument('--conf', default=0.15, type=float, help='overlap factor')
    parser.add_argument('--iou_thres', default=0.2, type=float, help='overlap factor')
    parser.add_argument('--flow', default=0.4, type=float, help='overlap factor')
    parser.add_argument('--velo', default=1, type=float, help='overlap factor')
    parser.add_argument('--mask', default=False, action='store_true')
    parser.add_argument('--omni', default=False, action='store_true')
    args = parser.parse_args()
    import os
    os.makedirs(os.path.split(args.output_path)[0], exist_ok=True)
    main(args)
    
