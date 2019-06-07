# ProbaV


## Installation
Clone this repo:
```
git clone --recurse-submodules https://github.com/loveorchids/probaV ~/Documents/sroie2019
```

## Requirement
Python:  3.5.2 or higher
```
pip install -r requirements.txt
```


## Prepare Data
1. Download dataset from: https://kelvins.esa.int/proba-v-super-resolution/home/
2. Create path ~/Pictures/dataset/SR/ and move downloaded dataset under it. <br />
Right now you under ~/Pictures/dataset/SR/ it should looks like:
```
SR
|--ProbaV
    |--train
    |--test
    |--norm.csv
```

### Training and Testing
Training, validation and test code are in one file. For each run, 
the program will split the train set into a training set and a validation set in a 9:1 proportion. 
And conduct 100 (default setting) training epoch, after each epoch validation will be performed to see
if the model is overfitted or not.
Finally, after training is completed, the model will test itself on the test set. <br />
```
cd ~/Documents/probaV/researches/img2img/probaV
python3 probaV_sr.py
```
When you start training, it will create a folder (name is probaV_exp) under: ~/Pictures/dataset/SR/<br />
Under this folder, there is
```
SR
|--ProbaV_exp
    |--grad  (discarded in this project)
    |--log  (discarded in this project)
    |--loss  (visualize the change of loss for every experiment)
    |--val  (validation result will be visualized here)
    |--%s_epoch_%d.pth  (%s: model's specific name assigned by user. %d: epoch number)
    |--%s_epoch_%d.pth  (%s: model's specific name assigned by user. %d: epoch number)
    |--%s_epoch_%d.pth  (%s: model's specific name assigned by user. %d: epoch number)
    |-- ......
```


### Code arrangement
under folder researches/img2img/probaV, the code arrangement is like below:
```
probaV_sr.py: the main program
pvsr_args.py: 
pvsr_augment.py: discarded, 
pvsr_data.py: 
pvsr_loss.py: 
pvsr_model.py: 
pvsr_module.py: 
pvsr_preprocess.py: 
pvsr_preset.py: 
```
