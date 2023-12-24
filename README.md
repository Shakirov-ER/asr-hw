# ASR HW1
#### Implemented by: Shakirov Emil

## Installation guide

Current repository is for Linux

(optional, not recommended) if you are trying to install it on macos run following before install:
```shell
make switch_to_macos
```

Then you run:

```shell
make install
```

## Run tests:

```shell
make test
```

## Download Language model:

```shell
make download_language_model
```
The file "lm.bin" will be in language_models/


## Download checkpoint:

```shell
make download_checkpoint
```
The file "model_best.pth" will be in default_test_model/

## Train model:

```shell
make train
```
Config for training you can find in hw_asr/config.json


## Test model:

### On test-clean:

```shell
make test_model_test_clean
```

The file "output_test_clean.json" with results will be in the root on repository

### On test-other:

```shell
make test_model_test_other
```

The file "output_test_other.json" with results will be in the root on repository


## Run any other python script:

If you want to run any other custom python script, you can just start it with "poetry run"
For example:

Instead of:

```shell
python test.py -r default_test_model/model_best.pth
```

You can use:

```shell
poetry run python test.py -r default_test_model/model_best.pth
```
