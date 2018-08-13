# White-Balance-By-Deep-learning

Pytorch implementation of [Ligeng Zhu](https://lzhu.me/) and [Brian Funt](http://www.cs.sfu.ca/~funt/) 's paper "Colorizing Color Images" (HVEI 2018)

[Torch implementation](https://github.com/Lyken17/Colorizing-Color-Images) by [Ligeng Zhu](https://lzhu.me/) 


## Demo

![demo](images/demos.png)


## Usage

### Training 
`python3 colorize.py train --dataset <dataset_dir> --save-model-name <model_name>`

```
usage: colorize.py train [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                         --dataset DATASET [--save-model-dir SAVE_MODEL_DIR]
                         [--save-model-name SAVE_MODEL_NAME]
                         [--image-size IMAGE_SIZE] [--cuda] [--seed SEED]
                         [--lr LR] [--log-interval LOG_INTERVAL]
                         [--checkpoint-dir CHECKPOINT_DIR] [--resume RESUME]
                         [--gpus [GPUS [GPUS ...]]]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of training epochs, default is 2
  --batch-size BATCH_SIZE
                        training batch size, default is 30
  --dataset DATASET     path to training dataset, the path should point to a
                        folder containing another folder with all the training
                        images
  --save-model-dir SAVE_MODEL_DIR
                        directory of the model to be saved, default is model/
  --save-model-name SAVE_MODEL_NAME
                        save model name
  --image-size IMAGE_SIZE
                        size of training images, default is 256
  --cuda                run on GPU
  --seed SEED           random seed for training
  --lr LR               learning rate, default is 0.001
  --log-interval LOG_INTERVAL
                        number of batches after which the training loss is
                        logged, default is 100
  --checkpoint-dir CHECKPOINT_DIR
                        checkpoint model saving directory
  --resume RESUME       resume training from saved model
  --gpus [GPUS [GPUS ...]]
                        specify GPUs to use

```

/<dataset_dir/> should be a directory containing images, for example mscoco train 2014 dataset.

Use `--resume` to resume from checkpoint

### Evaluating
`python3 colorize.py eval --input-dir <input_dir> --output-dir <output_dir> --model <model>`

```
usage: colorize.py eval [-h] --input-dir INPUT_DIR [--output-dir OUTPUT_DIR]
                        --model MODEL [--cuda] [--gpus [GPUS [GPUS ...]]]

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
                        path to input image directory
  --output-dir OUTPUT_DIR
                        path to output image directory
  --model MODEL         saved model to be used for evaluation
  --cuda                run on GPU
  --gpus [GPUS [GPUS ...]]
                        specify GPUs to use
```


--------

- To run on GPU, add `--cuda`

- To change other hyper parameters such as epochs, learning rate and batch size, use `python3 colorize.py {train | eval} -h` for details



## Reference  
[Fast Neural Style](https://github.com/pytorch/examples/tree/master/fast_neural_style) 