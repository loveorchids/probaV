# ProbaV
contest code for proba-V super resolution.

## Installation
How to clone this repo:
```
git clone --recurse-submodules https://github.com/loveorchids/probaV ~/Documents/probaV
```

## Requirement
Python:  3.5.2 or higher
```
pip install -r requirements.txt
```

## Prepare Data
1. [Download](https://kelvins.esa.int/proba-v-super-resolution/home/) dataset. 
2. Create path ```~/Pictures/dataset/SR/``` and move downloaded dataset under this folder. <br />
So under your ```~/Pictures/dataset/SR/``` folder it should looks like:
```
SR
|--ProbaV
    |--train
    |--test
    |--norm.csv
```

### Training and Testing
Training, validation and test code are in probaV_sr.py. For each run, 
the program will split the train set into a training set and a validation set in a 9:1 proportion. 
And conduct 100 (default setting) training epoch, after each epoch validation will be performed to see
if the model is overfitted or not.
Finally, after training is completed, the model will test itself on the test set. <br />
```
cd ~/Documents/probaV/researches/img2img/probaV
python3 probaV_sr.py --train --test
# if you only want to run thew train mode, then
python3 probaV_sr.py --train
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

### Trouble Shooting
In case the code does not execute properly:
1. check if you have cloned my other repository **omni_torch**. By default omni_torch should be 
located under your ```~/Documents/probaV/omni_torch```, and make sure it is not empty. 
```git clone https://github.com/loveorchids/omni_torch ~/Documents/probaV``` if omni_torch is empty.
2. I'm using a library called **imgaug** to perform image augmentation. Imgaug is still adding new 
exciting augmentation functions to make it better. If there is any error related to imgaug, please make
 sure you have its latest version. (**current 0.2.9**)
3. If there is any error related to **Matplotlib**, this usually happens when you have completed the training
 for several epoches. If such an error happens, this may related to your settings or tkinter installation. 
 You can search the errors on Google to find a solution to this.

### Report Content
- <a href='#Code-arrange'>Code Arrange</a>
- <a href='#Backgrounds'>Backgrounds</a>
- <a href='#Proposal'>Proposal</a>
- <a href='#Results'>Results</a>
- <a href='#Future-Work'>Future Work</a>
- <a href='#Reference'>Reference</a>

### Code Arrange
under folder researches/img2img/probaV, are the codes for super-resolution task:
```
probaV_sr.py: 
    the main program
pvsr_args.py: 
    some parameters needs to be passed to the program when using terminals
pvsr_data.py: 
    defines how to load data, rely on omni_torch
pvsr_loss.py: 
    defines a way to calculate loss, like cMSE, cPSNR introduced in
    https://kelvins.esa.int/proba-v-super-resolution/scoring/
pvsr_model.py: 
    definess several network architecture, 
    e.g. Residual Dense Network, CARN, and etc.
pvsr_module.py: 
    necessary module for network, and also include some other modules 
    for improving the result. e.g. self attention module, trellis module, etc.
pvsr_preprocess.py: 
    defines the way to load images and mask for the task. 
    In our implementation, we blended the mask and add to the original image with a alpha value.
pvsr_preset.py: 
    defines the settings and hyper-parameters for this task
```

## Backgrounds
Super Resolutions tasks could be divided into sinle-input super resolution (**SISR**) [3, 4] 
and multi-input super-resolutions (**MISR**). For **SISR** tasks, it has been well developed and 
extremely hard to achieve further improvement these days[1]. While more and more 
research interests were devoted into **MISR** tasks, e.g. reference-based super resolution
 (**RefSR**) [1, 2] and stereo image super resolution [6, 7]. <br>
For RefSR tasks, input data is a high resolution (**HR**) reference image has a similar content 
with the low resolution (**LR**) image. The key point to the task is to match the content 
from **HR** image to LR image efficiently. But in Proba-V task, we do not have
reference image in high resolution, thus methods for RefSR could hardly be applied into this project. <br>
For stereo image **SR** tasks, the key point to state-of-the-art methods would be a how 
to reduce the parallax. [7] tackle the problem by applying a modified version of Self-
Attention Module [8] to connect long-range spatial structure between two stereo images. Due 
to the all input images are captured from the same view point by the satellite, reducing 
the parallax or paying attention to the long-range spatial features may not help much 
in this project.<br>


## Proposal
#### Loading Data
In this project, the problem was tackled as an single-input super resolution task (though 
9-20 images will be fed to the network at each iteration). These 9-20 images were 
considered as a single multi-channel image. <br>
As all of the high resolution and low resolution images have a mask indicating clouds, and 
clouds are not required to be reconstruct. pixels with cloud are required to be eliminated.
But due to the mask are not optimal (usually they are bigger then the cloud area), see the
example below:<br>
![Data](https://github.com/loveorchids/probaV/blob/master/samples.png)<br>
So we blended the mask with the image by applying a gaussian filter with a random radius 
onto the mask and assign an alpha randomized from 0.1 to 0.4. Also, to make the contrast 
to be the same, an adaptive contrast normalization technique (**CLAHE**) was also applied.
Finally, to preserve the rich information, raw images (14-bit) loaded into our
model are extended into 16-bit depth, which proved to be better then 8-bit images. 
<br>

#### Network Architecture
To make a reasonable network, several network architecture has been made and tested.<by>
1. **Image-to-Image translation model:** <br>
At first, I tackle the problem to be an image-to-image translation in order to get a base line 
of this problem, the characteristic of image-to-image translation model was **down-convolution**
(encode information into semantic level information
to reduce the spatial size of feature map), **flat-convolution** (perform non-linear transformation 
to the feature map) and **up-convolution** (decode the semantic level into the spatial feature again).
2. **RDN [3]:** <br>
An implementation of vanilla Residual Dense Network.
3. **CARN [4]:** <br>
An implementation of vanilla Cascaded Residual Network.
4. **Trellis Module [5]:**<br>
Trellis Module was proposed by [5] in crowd density estimation, to encode the small and dense
object efficiently. We apply this module in order to encode the those low level feature effectively, so these
encoded information could be used for reconstruction.<br>
Due to the purpose of the Trellis module in [5] is different than the one in this project, some modifications 
are made.
5. **Self-Attention Module [8]:**<br>
Self-Attention module is designed to connect the long-range spatial feature, one could consider it as 
expanding the receptive field of the neural network to the whole image. In this project, it was designed 
to improve the result of of Image-to-Image translation model.
6. **Evaluator:**<br>
In Generative Adversarial Network (GAN), a discriminator is used to judge if the generated image satisfies 
the distribution of real data or not, in this project, a similar idea is used but more light-weight. 
We applied a VGG-16 model's convolutional part as a evaluator. After the super resolution model make 
its prediction, this prediction and ground truth will be sent to the evaluator to see if they have same 
"evaluation score". <br>
The "evaluation score" is defined as the MSE of prediction and ground truth of specific set of convolutional 
layers (e.g. conv1, conv2,  conv5).

#### Loss Functions and Optimizer
In this project, Loss functions are MSE, MAE, cMSE and cPSNR. cMSE and cPSNR are implemented 
according to the instructions in proba-V scoring page.<br>
As for optimizer, instead of Adam, we use Adabound [9] to carry out optimization. Adam is famous for its 
super convergence, but the result of Adam is usually not as optimal as SGD due to the unstable
and extreme learning rate. While Adabound is designed to compensate this problem by applying a 
dynamic bound on learning rates to avoid a violent oscillation of learning rate, and achieved a smoothed 
transition to SGD.

## Results
The network architecture which has the best numerical performance were RDN, RDN with trellis, 
followed by CARN, img2img translation model. 
The results of my projects were uploaded to 
[Google Drive](https://drive.google.com/file/d/1USPBeXBbmF1CtKrALtd7BzwZ2MnzBO4k/view?usp=sharing).
Numerical result comparison will be updated soon.

## Future Work
1. **Using Tensorflow to re-implement**<br>
current version of PyTorch usually tends to achieve sub-optimal result when compared to Tensorflow. 
I have also did an experiment about optimizing a model for cifar-10 tasks using a keras implementation
and a PyTorch implementation. Especially for tasks like super resolution, LSTM related models, sub-optimal 
results will cause large difference in the final result.
2. **NAS [10] for hyper-parameter tuning**<br>
Recently, "Learning to Learn" became a hot idea which combine the concept of Reinforcement Learning into tuning
the neural network structure. Support NAS for hyper-parameter tuning is also the goal for omni-torch.

## Reference
[1] Zhang, Zhifei, et al. "Image Super-Resolution by Neural Texture Transfer." arXiv preprint arXiv:
1903.00834 (2019).[link](https://arxiv.org/pdf/1903.00834.pdf)<br />
[2] Zheng, Haitian, et al. "CrossNet: An End-to-end Reference-based Super Resolution Network using 
Cross-scale Warping." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
[link](https://arxiv.org/pdf/1807.10547.pdf)<br />
[3] Zhang, Yulun, et al. "Residual dense network for image super-resolution." Proceedings of the IEEE 
Conference on Computer Vision and Pattern Recognition. 2018.[link](https://arxiv.org/pdf/1802.08797.pdf)<br />
[4] Ahn, Namhyuk, Byungkon Kang, and Kyung-Ah Sohn. "Fast, accurate, and lightweight super-resolution 
with cascading residual network." Proceedings of the European Conference on Computer Vision (ECCV). 
2018.[link](https://arxiv.org/pdf/1803.08664.pdf)<br />
[5] Jiang, Xiaolong, et al. "Crowd Counting and Density Estimation by Trellis Encoder-Decoder Network." 
arXiv preprint arXiv:1903.00853 (2019).[link](https://arxiv.org/pdf/1903.00853.pdf)<br />
[6] Jeon, Daniel S., et al. "Enhancing the spatial resolution of stereo images using a parallax prior." 
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
[link](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jeon_Enhancing_the_Spatial_CVPR_2018_paper.pdf)<br />
[7] Wang, Longguang, et al. "Learning parallax attention for stereo image super-resolution." 
arXiv preprint arXiv:1903.05784 (2019).[link](https://arxiv.org/pdf/1903.05784.pdf)<br />
[8] Zhang, Han, et al. "Self-attention generative adversarial networks." arXiv preprint arXiv:1805.08318 
(2018). [link](https://arxiv.org/pdf/1805.08318.pdf) <br>
[9] Luo, Liangchen, et al. "Adaptive gradient methods with dynamic bound of learning rate." arXiv 
preprint arXiv:1902.09843 (2019). [link](https://arxiv.org/abs/1902.09843.pdf)<br>
[10] Zoph, Barret, and Quoc V. Le. "Neural architecture search with reinforcement learning." arXiv 
preprint arXiv:1611.01578 (2016). [link](https://arxiv.org/abs/1611.01578)
