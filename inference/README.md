# Inference Docker Build
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