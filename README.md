# All-Cell Segmenter: An All-purpose Framework for Multi-modality Cell Segmentation
This repository is the official implementation of [All-Cell Segmenter: An All-purpose Framework for Multi-modality Cell Segmentation](https://arxiv.org).

## Environments and Requirements
The framework has been verified in the following environments:

+ System: Ubuntu 18.04.5 LTS
+ CPU: Intel(R) Xeon(R) CPU E5-2620 v4 @2.10GHz
+ RAM: 128GiB
+ GPU: Nvidia GeForce RTX3090 24G x2
+ CUDA Version: 11.3
+ Python 3.9

To install requirements:
```
pip install -r requirements.txt
```
## Dataset
The competition dataset can be downloaded in [Official Website](https://neurips22-cellseg.grand-challenge.org/).
Unzip all archives and structure datasets as the followings:
```
-- NeurIPS_CellSegData
    -- Train_Labeled
        -- images
            -- cell_00001.bmp
            -- cell_00002.bmp
            -- ...
            -- cell_01000.png
        -- labels
            -- cell_00001_label.tiff
            -- cell_00002_label.tiff
            -- ...
            -- cell_01000_label.tiff
    -- Train_Unlabeled
        -- images # The preprocessed unlabeled images will be copied to this folder.
        -- labels # The pseudo label
        -- release-part1
            -- unlabeled_cell_00000.png
        -- release-part2-whole-slide
            -- whole_slide_00001.tiff
            -- ...
    -- TuningSet
        -- cell_00001.tiff
        -- ...
```


## Training
We train the full framework with the following processes.

### Backbone Weights
`convertCkpt.py` is used for converting the weights of ImageNet-pretrained ResNet-18 from torchvision. In the contest, we use [this weights](https://download.pytorch.org/models/resnet18-5c106cde.pth) instead. You can modify `convertCkpt.py` to convert the above weights. (Actually, we find the two weights is numerically same.)
Run `convertCkpt.py` with command and you will get a pretrained backbone used in our model:
```
python convertCkpt.py
```
### Anchors
The detection based watershed branch used in our framework (Anchor-based Watershed Framework) needs predefined anchor size, which is obtained from [autoanchor](utils/autoanchor.py) with KMeans algorithm. However, due to refraction during competition, we're sorry that this script cannot run anymore. **We'll update code if we find the correct way to reproduce kmeans results.** We adopt three anchors in our model: `[22, 22], [46, 46], [199, 190]`, which can support segmentation for cells with `11 ~ 400`px bounding box size.

### Baseline Training
First of all, train the baseline model with following command
```
torchrun --nproc_per_node 2 train_main.py \
    --img 640 --batch 32 --epochs 300 \
    --data data/cellseg/labeled.yaml   --workers 16 --name fulltrain \
    --noautoanchor --save-period 10 --weights resnet18.pt   \
    --patience 0 --hyp data/hyps/hyp.labeled.yaml --device 0,1 \
    --cos-lr --sync-bn

torchrun --nproc_per_node 2 train_main.py \
    --img 640 --batch 32 --epochs 50 \
    --data data/cellseg/labeled.yaml   --workers 16 --name multiscale \
    --noautoanchor --save-period 5 --weights fulltrain.pt   \
    --patience 0 --hyp data/hyps/hyp.labeled_multiscale.yaml --device 0,1 \
    --cos-lr --sync-bn
```

### Pseudo Label generation
Before weakly-supervised finetuning, we generate pseudo label for all unlabeled data:

```
python predict.py -i path_to_cropped_unlabeled_data \
                -o output_path \
                --ckpt finetuned_baseline.pt
```

### Optimization for each modality
Then finetune AWF with following commands:
```
torchrun --nproc_per_node 2 train_weakly.py \
    --img 640 --batch 32 --epochs 50 \
    --data data/cellseg/weakly_{bf/fl/gs}.yaml   --workers 16 --name weakly_{bf/fl/gs} \
    --noautoanchor --save-period 5 --weights multiscale.pt   \
    --patience 0 --hyp data/hyps/hyp.ws_{general/flourescene/grayscale}.yaml \
    --cos-lr --sync-bn

```

### Omnipose Training
The Omnipose used in our framework is slightly different with [official verison](https://github.com/kevinjohncutler/omnipose) for compatibility. We simply extract essential methods from official implementation and rewrite it in PyTorch. We train Omnipose with following command:
```
torchrun --nproc_per_node 2 train_omnipose.py --img 640 --batch 32 --epochs 100 \
    --data data/omnipose.yaml  --workers 8   --cos-lr \
    --name omnipose_reproduce --noautoanchor --save-period 5 \
    --weights baseline.pt --patience 0


torchrun --nproc_per_node 2 train_omnipose.py --img 640 --batch 32 --epochs 100 \
    --data data/omnipose.yaml  --workers 3   --cos-lr \
    --name omnipose_reproduce --noautoanchor --save-period 5 \
    --weights runs/train/multiscale_reproduce7/weights/best.pt --patience 0
```
We spent lots of time to debug the code of Omnipose, but there may still be some bugs due to inferior performance.

## Trained Models
You can download trained models [here](https://drive.google.com/drive/folders/1yz9Gb4Y5LPfk5DOi96G77YMqWnQrnUKf?usp=sharing).



## Evaluation
Evaluate the checkpoint on any labeled dataset you want. Performance for each image will be saved to `-o` parameter.
```
python evaluate.py -i path_to_input_data \
                    -o evalute_detail.csv \
                    --ckpt path_to_checkpoint.pt




```
We've manually finetuned NMS parameters in `predict.py`, if you want to try your own, you can modify line `120-122` by yourself. If you want to evaluate docker performance, please build docker first, then run the following command over the docker prediction.

```
python compute_metric.py --gt_path path_to _ground_truth \
                        --seg_path path_to_prediction 
```



## Docker Build
After pipeline training, copy all branch checkpoints to `ckpts` folder, then modify line `144-149` of  `predict.py` to select correct branch for each modality.
The default settings of varaible `model_dict` is as follows:

```
    model_dict = {
        'bf': 'general.pt',  # brightfield branch
        'gs': 'grayscale.pt', # grayscale branch
        'fl': 'fl.pt',  # flourescence branch
        'omni': 'omnipose.pt' # omnipose model
    }
```

Then you can predict masks with the following command:

```
python predict.py -i "path_to_inputs"  -o "path_to_outputs"
```

If the program runs correctly, build and save docker image with following commands:

```
docker build -t name:tag .
```

```
docker save name:tag -o name.tar.gz
```

## Results

Our method achieves the following performance on [Weakly Supervised Cell Segmentation Contest](https://neurips22-cellseg.grand-challenge.org/evaluation/challenge/leaderboard/).


| Model | Val F1-Score | Test F1-Score | 
|  ----  |  ----  | ---- |
| Cell Segmenter | 0.8537 | |

# Contributing

The repository is mainly modified from [yolov5](https://github.com/ultralytics/yolov5) official implementation and follows GPL-3.0 license.


