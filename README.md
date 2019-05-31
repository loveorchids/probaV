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

### Training
```
# Run in Terminal
cd ~/Documents/probaV/researches/img2img/probaV
python3 probaV_sr.py
```


### Testing
```
# Run in Terminal
cd ~/Documents/probaV/researches/img2img/probaV
python3 probaV_sr.py --test
```
