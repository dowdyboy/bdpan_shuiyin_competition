# 百度网盘AI大赛 -去水印模型冲刺赛第1名方案
> 这是一个基于PaddlePaddle的图像去水印的解决方案，本方案在B榜上取得了第1名的成绩，本文将介绍本方案的一些细节，以及如何使用本方案进行预测。

## 项目描述
图像去水印是一个低层次生成式任务，其目标是从带水印的图像中恢复出原始图像。
本项目采用了基于CNN的图像去水印模型，并结合了混合注意力机制，很大程度提升了模型的表达能力。
同时，在数据处理上，采用了多种数据增强方案，防止过拟合，提升模型的泛化能力。

## 项目结构
```
-|bdpan_shuiyin
-|checkpoints
-|data
-|dowdyboy_lib
-|scripts
-train_ms.py
-train_ms_s2.py
-predict.py
-predict_single.py
```
- bdpan_shuiyin: 本项目的模型源代码
- checkpoints: 本项目的模型参数
- data: 本项目的数据集
- dowdyboy_lib: 自行编写的基于飞桨的深度学习训练器，详见[这里](https://github.com/dowdyboy/dowdyboy_lib)
- scripts: 本项目的历史版本
- train_ms.py: 训练脚本，使用多尺度训练
- train_ms_s2.py: 训练脚本，使用多尺度训练，使用psnr作为损失函数
- predict.py: 预测脚本，使用测试时增强
- predict_single.py: 预测脚本，不使用测试时增强

## 数据

本项目测试数据由百度网盘AI大赛提供，详见[官网](https://aistudio.baidu.com/aistudio/competition/detail/706/0/datasets) 。

训练验证数据 下载：
链接：https://pan.baidu.com/s/1PMmOPzRQTEcVrur68PUXxQ?pwd=2ngu
提取码：2ngu

将训练数据解压到data文件夹下，解压后的文件夹结构如下：
```
|data
-|bdpan_shuiyin
--|train
--|val
--|test_make
--|test_real
```

## 训练
> 单卡训练
```bash
# 多尺度训练，使用l1作为损失函数
python ./train_ms.py 
        --train-data-dir data/bdpan_shuiyin/train/ 
        --val-data-dir data/bdpan_shuiyin/val/ 
        --num-workers 8 
        --use-scheduler 
        --batch-size 2 
        --out-dir ./output_ms_v7 
        
# 多尺度训练，使用psnr作为损失函数，降低学习率
python ./train_ms_s2.py 
        --train-data-dir data/bdpan_shuiyin/train/ 
        --val-data-dir data/bdpan_shuiyin/val/ 
        --num-workers 8 
        --use-scheduler 
        --batch-size 2 
        --out-dir ./output_ms_v7_s2 
        --resume {第一阶段的最优模型的路径}
```
> 多卡训练
```bash
# 多尺度训练，使用l1作为损失函数
python -m paddle.distributed.launch ./train_ms.py 
        --train-data-dir data/bdpan_shuiyin/train/ 
        --val-data-dir data/bdpan_shuiyin/val/ 
        --num-workers 8 
        --use-scheduler 
        --batch-size 2 
        --out-dir ./output_ms_v7 
        --sync-bn 
        --multi-gpu
        
# 多尺度训练，使用psnr作为损失函数，降低学习率
python -m paddle.distributed.launch ./train_ms_s2.py 
        --train-data-dir data/bdpan_shuiyin/train/ 
        --val-data-dir data/bdpan_shuiyin/val/ 
        --num-workers 8 
        --use-scheduler 
        --batch-size 2 
        --out-dir ./output_ms_v7_s2 
        --resume {第一阶段的最优模型的路径}
        --sync-bn 
        --multi-gpu
```
> 竞赛时，我们使用了2卡NVIDIA A100 80G进行了训练；

## 预测
> 运行predict.py脚本，采用了测试时增强，精度更高
```bash
python predict.py 
     <要预测的图像文件夹路径> <预测结果文件夹路径>
```
> 运行predict_single.py脚本，不使用测试时增强，精度高的同时速度更快
```bash
python predict_single.py 
     <要预测的图像文件夹路径> <预测结果文件夹路径>
```

## 项目详情

### 数据处理

由于训练数据并非纯粹的静态数据，需要自己生成，因此我们决定采用动态数据加载的方式，即，在训练脚本运行过程中，动态的每次生成不一样的训练数据，通过这种方式可以很大程度提升模型的泛化能力。

在生成的水印类别方面，我们的数据生成器可以生成三种类别的带水印图像：
（1）基于官网水印生成脚本的文字水印；
（2）基于官网真实水印的掩码生成的图形水印；
（3）基于真实场景图像生成的图形水印；

我们通过配置文件，为不同的水印类别配置不同的概率。

对于图像预处理部分，我们采用了多种类型的数据增强方案，包括：
（1）水平翻转、垂直翻转
（2）旋转
（3）光度变换
（4）大尺寸cutout、小尺寸cutout
（5）椒盐噪声

我们针对弱数据增强配置较高的概率，针对强数据增强配置较低的概率，这样能够在维持数据总体分布的基础上尽可能的提升模型性能和泛化性。数据增强的参考论文有：http://arxiv.org/abs/1710.09412 。

另外，我们使用了多尺度训练，在对输入图像进行裁切时，采用了不同的尺度：512, 640, 768, 896, 1024, 1152, 1280, 1536。这种方式能够使模型尽可能适应不同大小的输入图像，通过提取多种不同尺度特征图的方式提升模型性能。

### 网络设计

1.总体架构

我们的方案采用了基于EarseNet的网络架构。其论文为：https://ieeexplore.ieee.org/document/9180003 。

EraseNet由生成器和判别器组成，计算量主要在生成器部分。这个网络采用了类似于UNET的结构做粗处理，并且新颖的是，在粗处理部分，引入和分割任务作为辅助训练，分割任务的GT只需要对两张图片进行比对即可。擦除网络的粗处理结束后，进行细处理部分，这一部分使用了空洞卷积来使网络能够捕获更多的捕获全局信息，生成更清晰的图片。

由于EraseNet原始论文中所面对的是复杂的自然场景环境，而本竞赛只作用在文档的文字擦除环境中。因此，考虑到训练成本和收敛的难易程度，我们放弃了原始EraseNet的判别器部分，只采用了生成器，并且移除了分割头。

2.强化Refinement模块

原始EarseNet的强化部分只采用了基本的跳连结构，并且层数较浅。这样可能导致图像还原在细节上不够充分。由于文档图像有大量文字，对细节还原要求更高。因此，我们采用了更深层次的UNET网络作为Refinement模块。

另外，为了更好的捕获全局特征，我们还引入了NonLocalBlock，将其插入在网络的下采样部分。NonLocalBlock是一种用于图像和视频中的深度神经网络的模块，旨在捕捉远距离依赖关系，可以用于提高模型的感知和理解能力。在传统的卷积神经网络中，每个神经元的输出只取决于它在输入图像中的局部感受野，这可能无法有效地捕捉到像素之间的全局相互作用和长距离依赖性。相比之下，NonLocalBlock可以让每个神经元对整个输入特征图进行自注意力机制的计算，从而引入全局上下文信息，使得神经元的输出不仅取决于它的局部感受野，还考虑了整个输入特征图的全局信息。这使得NonLocalBlock在处理图像中的长距离依赖关系、跨通道信息的相互作用等方面表现出了非常出色的性能。

3.混合注意力

为了更好的提升网络的表达能力，我们在网络的输出卷积之前添加了混合注意力模块。该模块同时使用了像素注意力和通道注意力，能够在更多样的维度上让网络关注更具备有效内容的区域。这样能够更好的恢复文档图像中带有文本内容的区域。

### 训练方案

1.第一阶段

我们使用Adam优化器，并使用CosineAnnealingRestartLR调整学习率，初始学习率设置为2e-4。损失函数使用L1函数。 我们在全量数据集上进行500个epoch的训练，并在每个epoch的学习结束后，在验证集上进行验证，计算psnr和ssim，最终输出模型为在验证集上表现最佳的模型。

2.第二阶段

选取第一阶段的最优模型作为预训练模型。使用Adam优化器，并使用CosineAnnealingRestartLR调整学习率，初始学习率设置为2e-5。损失函数使用PSNR函数。在全量数据集上进行200个epoch的训练。在验证集上进行验证，计算psnr和ssim，最终输出模型为在验证集上表现最佳的模型。

我们的整个实验过程在2张NVIDIA A100 80G GPU上进行。

