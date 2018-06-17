# White-Balance-By-Deep-learning

Pytorch implementation of [Ligeng Zhu](https://lzhu.me/) and [Brian Funt](http://www.cs.sfu.ca/~funt/) 's paper "Colorizing Color Images" (HVEI 2018)

[Torch implementation](https://github.com/Lyken17/Colorizing-Color-Images) by [Ligeng Zhu](https://lzhu.me/) 


## Demo

![demo](images/demos.png)


## Usage

### Training 
`python3 colorize.py train --dataset <dataset_dir> --save-model-name <model_name>`

- Dataset is a directory containing all white balanced training images

### Evaluating
`python3 colorize.py eval --input-dir <input_dir> --output-dir <output_dir> --model <model>`

- Save transformed image in output dir

--------

To run on GPU, add `--cuda`

To change other hyper parameters such as epochs, learning rate and batch size, use `python3 colorize.py {train | eval} -h` for details



## Reference  
[Fast Neural Style](https://github.com/pytorch/examples/tree/master/fast_neural_style) 