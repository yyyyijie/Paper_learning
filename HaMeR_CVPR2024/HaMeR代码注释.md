# HaMeR: Hand Mesh Recovery

**Reconstructing Hands in 3D with Transformers**

[Georgios Pavlakos](https://geopavlakos.github.io/), [Dandan Shan](https://ddshan.github.io/), [Ilija Radosavovic](https://people.eecs.berkeley.edu/~ilija/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/), [David Fouhey](https://cs.nyu.edu/~fouhey/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)

[![arXiv](https://img.shields.io/badge/arXiv-2312.05251-00ff00.svg)](https://arxiv.org/pdf/2312.05251.pdf)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://geopavlakos.github.io/hamer/)     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rQbQzegFWGVOm1n1d-S6koOWDo7F2ucu?usp=sharing)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/geopavlakos/HaMeR)



## 论文

1.Introduction

本节介绍中概述了研究背景和技术挑战，指出自动感知3D手部姿态对于理解人与环境互动的重要性。最近的研究趋势显示，通过使用简单的高容量模型和大量数据可以取得进展。本研究提出了一种名为HaMeR的新方法，它是一种基于Transformer的单帧手部网格恢复方法，能够在多种条件下准确地重建3D手部姿态。

2.Related work

回顾了3D手部姿态和形状估计领域的历史发展和主要进展，包括参数化和非参数化方法，以及针对特定挑战（如遮挡、速度、概率建模和运动模糊）的专门解决方案。HaMeR方法与这些方法的区别在于它采用了简单的设计，并且重点放在扩展训练数据的规模和模型的容量上，以提高手部网格恢复的准确性。

3.1. MANO Parametric Hand Model

介绍了所采用的人手参数化模型MANO，它接受手部姿态参数θ和形状参数β作为输入，并定义了一个函数M(θ, β)，返回具有V=778个顶点的手部网格M。此外，MANO还会返回手部的K=21个关节的位置X。

3.2. Hand Mesh Recovery

主要说明了给定一张手部的RGB图像I时，目标是重建3D手部表面。通过估计图像中手部的MANO姿态和形状参数来解决这个问题。类似于之前的参数化人体和手部重建工作，使用一个网络来学习从图像像素到MANO参数的映射f，并同时估计相机参数π。

3.3. Architecture

本节描述了HaMeR的体系结构，采用了完全基于Transformer的设计，使用Vision Transformer (ViT)作为骨干网，后面跟着一个Transformer解码器来回归手部和相机参数。输入的RGB图像首先被转换为图像块，然后作为输入令牌送入ViT。ViT处理图像块并返回一系列输出令牌。Transformer解码器则处理单个令牌，并利用ViT输出的令牌进行交叉注意力处理，最终输出回归的手部参数

3.4. Losses

训练损失结合了2D和3D损失，以监督模型的学习。对于提供3D地面真值的数据，可以直接应用于模型参数的损失。对于只提供2D注解的数据，使用重投影损失来确保预测的关键点与2D注解的一致性。

4.HInt: Hand Interactions in the wild

介绍了HInt数据集，这是一个用于评估和训练手部姿态估计系统的新型数据集，特别关注自然环境中的手部交互。HInt数据集是从三个视频数据集构建而成的，分别是Hands23、Epic-Kitchens和Ego4D，包含YouTube视频中的日常活动片段，烹饪行为的VISOR数据集中的帧，和手部与物体的交互时刻。该数据集总共包含40,400个手部注解，其中New Days有12,000个，VISOR有5,300个，Ego4D有23,200个。这些注解不仅包括2D关键点位置，还包括“存在”和“遮挡”标签，以指示关键点是否存在于图像帧内以及是否被遮挡。

5.1. 3D pose accuracy

报告了HaMeR在标准3D手部姿态基准测试上的定量结果，包括FreiHAND和HO3Dv2数据集。HaMeR在这些数据集上的3D关节和3D网格精度等指标均优于之前的工作。

5.2. 2D pose accuracy

展示了HaMeR在野外条件下（即非实验室环境）的数据集上的表现，使用了HInt数据集进行评估。HInt包含2D手部关键点注释，用于评估3D方法的2D重投影精度，HaMeR在这个数据集上也表现出色。

5.3. Ablation Analysis

进行了消融实验来分析不同设计选择对系统性能的影响。实验结果显示，增加训练数据的规模和使用更大容量的模型可以带来更好的性能。

## 代码
demo.py
```
# 导入所需的库
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

# 从HAMeR包中导入所需的模块和配置
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

# 定义颜色常量，用于渲染
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

# 导入自定义的ViTPoseModel类
from vitpose_model import ViTPoseModel

# 导入json库和typing库，用于处理JSON数据和类型注释
import json
from typing import Dict, Optional
```

接下来进入main函数，对处理命令行参数
```bash
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()
```

部分参数说明：

--checkpoint，默认从./_DATA/hamer_ckpts/checkpoints/hamer.ckpt获取，里面是从https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz下载的预训练模型

--side_view，默认为False，如果设置该参数，还会渲染侧面视图

--full_frame，默认为True，如果设置该参数，将渲染所有人物的组合视图

--rescale_factor，默认边界框缩放因子为 2.0，用来指定边界框的缩放因子

--body_detector，默认使用 'vitdet' 作为身体检测器，限制参数值只能是 'vitdet' 或 'regnety'，使用 'regnety' 可以提高运行速度并减少内存使用



```bash
# 下载并加载预训练模型检查点
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

```

load_hamer()在hamer/hamer/models/__init__.py里面有定义，该函数输入是预训练模型路径，返回值是预训练的HAMeR模型，以及它的配置
```bash
def load_hamer(checkpoint_path=DEFAULT_CHECKPOINT):
    from pathlib import Path
    from ..configs import get_config
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True) #使用 get_config 函数加载模型配置，并设置 update_cachedir=True，用于更新配置文件中缓存目录的路径。

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()  #使用 defrost 方法取消模型配置的冻结状态，允许对配置进行修改
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256] #设置 BBOX_SHAPE 为 [192, 256]，这可能是为了确保边界框的尺寸与模型期望的尺寸一致
        model_cfg.freeze()  #使用 freeze 方法重新冻结配置，防止进一步修改。

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):  #检查配置文件中是否指定了预训练权重
        model_cfg.defrost()  #取消配置冻结
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')  #移除 PRETRAINED_WEIGHTS 项
        model_cfg.freeze()  #重新冻结配置

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg
```

回到demo.py文件
```bash
# 设置HAMeR模型
    # 根据是否有CUDA支持，选择使用CPU或GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)  # 将模型移动到选择的设备上
    model.eval()  # 将模型设置为评估模式
    
     # 加载人体检测器
    # 根据参数选择使用哪种人体检测器
    if args.body_detector == 'vitdet':
        # 使用detectron2的LazyConfig加载配置文件
        from detectron2.config import LazyConfig
        import hamer
        # 设置配置文件路径
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        # 设置检测器使用的预训练模型检查点
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        # 设置检测器的测试得分阈值
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        # 创建一个Lazy Predictor对象
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        # 使用detectron2的model_zoo和get_config获取配置
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        # 获取regnety模型的配置
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        # 设置检测器的测试得分阈值和NMS阈值
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
```

这段代码的作用是设置和初始化一个人体检测器，它是 HaMeR 系统中的一个组件，用于检测图像中的人体位置。首先，代码从 `hamer.utils.utils_detectron2` 模块中导入了 `DefaultPredictor_Lazy` 类，这是一个用于人体检测的类。根据用户通过命令行参数 `body_detector` 选择的检测器类型

如果选择的是 `'vitdet'` 检测器：设置配置文件路径 `cfg_path`，指向configs路径下一个 Cascade Mask R-CNN 的配置文件，该模型使用了 Vision Transformer 作为骨干网络。加载配置文件并创建配置对象 `detectron2_cfg`，设置模型的初始化检查点来自https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl，调整模型的测试得分阈值 `test_score_thresh`，用于确定预测框是否被认为是正样本。这里将前三个阶段的得分阈值都设置为 0.25。

如果选择的是 `'regnety'` 检测器：获取一个使用 RegNetY 作为骨干网络的 Mask R-CNN 模型的配置，调整模型的测试得分阈值 `test_score_thresh` 和非极大值抑制阈值 `test_nms_thresh`，分别设置为 0.5 和 0.4

最后，使用 `detectron2_cfg` 配置对象创建一个 `DefaultPredictor_Lazy` 实例，这个实例将用于实际的人体检测任务。

从hamer/hamer/utils/utils_detectron2.py可以查看 `DefaultPredictor_Lazy`的说明：

Create a simple end-to-end predictor with the given config that runs on single device for a single input image.

Compared to using the model directly, this class does the following additions:

\1. Load checkpoint from the weights specified in config (cfg.MODEL.WEIGHTS).

\2. Always take BGR image as the input and apply format conversion internally.

\3. Apply resizing defined by the config (`cfg.INPUT.{MIN,MAX}_SIZE_TEST`).

\4. Take one input image and produce a single output, instead of a batch.

该函数通过输入配置文件和图像（BGR格式），输出instance字典，包括人体预测框和对应的分数



```
    # keypoint detector
    cpm = ViTPoseModel(device)   #初始化了一个关键点检测器 cpm，使用 ViTPoseModel 类创建。

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces) #设置了渲染器 renderer，使用 Renderer 类创建。

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)  #创建输出目录

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]   #获取所有演示图像路径
```

`ViTPoseModel` 是一个用于检测人体姿态的关键点检测模型，**renderer**渲染器将用于将三维手部网格渲染到二维图像上。`model_cfg` 参数包含了模型的配置信息，`model.mano.faces` 提供了用于渲染的三维网格的面信息。`ViTPoseModel`在hamer/vitpose_model.py中有相关构造函数如下：

```
class ViTPoseModel(object):
    # 模型配置字典，包含不同模型的配置信息和权重文件路径
    MODEL_DICT = {
        'ViTPose+-G (multi-task train, COCO)': {
            'config': f'{VIT_DIR}/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py',
            # 配置文件的路径，其中包含模型结构、超参数等信息
            'model': f'{ROOT_DIR}/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth',
            # 预训练模型权重文件的路径
        },
    }

    def __init__(self, device: str | torch.device):
        # 初始化ViTPoseModel对象
        self.device = torch.device(device)
        # 将输入设备转换为torch.device对象（如'cuda:0'或'cpu'）
        self.model_name = 'ViTPose+-G (multi-task train, COCO)'
        # 设置当前使用的模型名称
        self.model = self._load_model(self.model_name)
        # 加载指定的模型

    def _load_model(self, name: str) -> nn.Module:
        # 加载指定名称的模型
        dic = self.MODEL_DICT[name]
        # 根据模型名称从字典中获取配置信息和权重路径
        ckpt_path = dic['model']
        # 获取权重文件的路径
        model = init_pose_model(dic['config'], ckpt_path, device=self.device)
        # 使用配置文件和权重文件初始化模型
        return model
        # 返回初始化好的模型
```

继续回到demo.py

```
 # 遍历文件夹中的所有图像
    for img_path in img_paths:
        # 使用OpenCV读取图像
        img_cv2 = cv2.imread(str(img_path))

        # 检测图像中的人体
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]  # RGB到BGR格式转换

        # 获取检测到的实例
        det_instances = det_out['instances']
        # 过滤出分数高于0.5的实例，并且是人类（类别0）
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        # 获取预测的边界框和得分
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # 对每个人进行关键点检测
        # predict_pose函数输入图像和边界框及得分信息，输出关键点检测结果
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # 初始化用于存储边界框和是否为右手的列表
        bboxes = []
        is_right = []

        # 根据手部关键点检测结果，使用手部边界框和左右手标记
        for vitposes in vitposes_out:
            # 获取左手和右手的关键点
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # 过滤出置信度高于0.5的关键点，并计算边界框
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:     #置信度高的关键点大于3个认为有效
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)  # 左手

            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)  # 右手

        # 如果没有检测到手，则跳过当前图像
        if len(bboxes) == 0:
            continue

        # 将边界框堆叠成numpy数组
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
```

代码首先读取图像进行人体检测，筛选出分类为人体（`pred_classes == 0`）且置信度大于 0.5 的预测框和对应的分数的实例，再使用初始化的关键点检测器 `cpm` 对图像中的每个人进行关键点检测，`predict_pose` 函数接受图像和预测框及其置信度作为输入。接下来遍历每个人的关键点检测结果，提取左右手的关键点，并根据关键点的置信度（`keyp[:,2] > 0.5`）确定是否接受检测结果。如果检测到足够多（`sum(valid) > 3`）的置信度高的关键点，则生成手部的边界框 `bbox` 并记录是左手还是右手（`is_right.append(0)` 或 `is_right.append(1)`）。获取左手和右手的关键点时数字 `-42` 和 `-21` 是基于关键点数组中预定义的索引，它们表示关键点的起始位置。

接下来的代码是对每只手的三维重建，并将重建结果渲染到二维图像中，同时提供了保存重建的手部网格数据的功能。

```
        # 运行重建，对所有检测到的手进行处理
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        # 创建数据加载器，用于批量处理
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        # 初始化存储顶点和相机参数的列表
        all_verts = []
        all_cam_t = []
        all_right = []
        
        # 遍历数据加载器中的所有批次
        for batch in dataloader:
            # 将批次数据移动到设备上
            batch = recursive_to(batch, device)
        	with torch.no_grad():
            	out = model(batch)

        	# 根据左右手调整相机参数
        	multiplier = (2 * batch['right'] - 1)
       		pred_cam = out['pred_cam']
        	pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        	# 计算相机参数的全尺寸转换
        	box_center = batch["box_center"].float()
        	box_size = batch["box_size"].float()
        	img_size = batch["img_size"].float()
        	scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        	pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        	# 渲染结果
        	batch_size = batch['img'].shape[0]
        	for n in range(batch_size):
            	# 从图像路径获取文件名
            	img_fn, _ = os.path.splitext(os.path.basename(img_path))
            	person_id = int(batch['personid'][n])

            	# 将图像数据转换为渲染器所需的格式
            	white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (DEFAULT_STD[:, None, None] / 255)
            	input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (DEFAULT_MEAN[:, None, None] / 255)
            	input_patch = input_patch.permute(1, 2, 0).numpy()

            	# 调用渲染器，渲染手部网格模型
            	regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                    out['pred_cam_t'][n].detach().cpu().numpy(),
                                    batch['img'][n],
                                    mesh_base_color=LIGHT_BLUE,
                                    scene_bg_color=(1, 1, 1),
                                    )


```

这段代码先使用 `ViTDetDataset` 创建一个数据集对象，该对象包含了用于重建的手部边界框信息和其他必要配置，有关 `ViTDetDataset`存放在hamer/hamer/datasets/vitdet_dataset.py文件中可以找到如下。然后，使用 `torch.utils.data.DataLoader` 创建一个数据加载器，设置batchsize为8，不打乱数据，以便批量处理。将数据转移到指定的设备（GPU或CPU），在不计算梯度的情况下进行模型的前向传播，获取重建结果。再根据左右手标识调整预测的相机参数，以适应左右手的视角差异，将相机参数从裁剪图像尺寸转换为原始图像尺寸。对于每个批次中的每张图像，使用 `renderer` 对象渲染三维手部网格，并根据需要渲染侧面视图，renderer的构造函数位于hamer/hamer/utils/renderer.py

```
#hamer/hamer/datasets/vitdet_dataset.py
# 定义了用于图像标准化的默认均值和标准差，
# 这些值用于将图像数据转换到 [0,1] 范围内，并进行标准化处理。
DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
class ViTDetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 img_cv2: np.array,
                 boxes: np.array,
                 right: np.array,
                 rescale_factor=2.5,
                 train: bool = False,
                 **kwargs):
        super().__init__()  # 调用基类的构造函数
        self.cfg = cfg  # 模型配置，包含网络参数和数据相关的配置信息
        self.img_cv2 = img_cv2  # 原始图像数据，使用 OpenCV 格式（BGR）
        # self.boxes = boxes 

        assert train == False, "ViTDetDataset is only for inference"  # 断言确保这个数据集只用于推理（非训练模式）
        self.train = train  # 是否在训练模式下，这里固定为 False
        self.img_size = cfg.MODEL.IMAGE_SIZE  # 根据配置获取图像尺寸
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)  # 根据配置获取图像的均值，并乘以 255（转换到像素值）
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)  # 根据配置获取图像的标准差，并乘以 255

        # Preprocess annotations  # 开始预处理标注信息
        boxes = boxes.astype(np.float32)  # 确保边界框数据是浮点数类型

        # 计算边界框的中心点
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        # 计算边界框的尺度，基于边界框宽度和高度，乘以 rescale_factor 进行缩放
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        # 创建一个表示每个人唯一标识的数组
        self.personid = np.arange(len(boxes), dtype=np.int32)
        # 将表示左右手的数组转换为浮点数类型
        self.right = right.astype(np.float32)
```

vitdet_dataset.py中这个类的构造函数接收图像数据、边界框、左右手标识和一些其他的配置参数。它首先确保数据集不是在训练模式下使用（因为 `ViTDetDataset` 仅用于推理）。然后，它从配置中获取了一些用于数据预处理的参数，比如图像的均值和标准差。接下来，它计算了边界框的中心点和尺度，这些将用于后续的网络输入标准化。最后，它创建了 `personid` 数组和处理了 `right` 数组，这些信息将用于标识图像中每个人的唯一性和手的左右标识。

继续demo.py

```
				# 如果需要渲染侧面视图
            	if args.side_view:
                	side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                    out['pred_cam_t'][n].detach().cpu().numpy(),
                                    white_img,
                                    mesh_base_color=LIGHT_BLUE,
                                    scene_bg_color=(1, 1, 1),
                                    side_view=True)
                	# 将原图、渲染图和侧面图拼接在一起
                	final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
            	else:
                	# 只拼接原图和渲染图
                	final_img = np.concatenate([input_patch, regression_img], axis=1)

            	# 将渲染后的图像写入到文件
            	cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255 * final_img[:, :, ::-1])

            	# 将顶点和相机参数添加到列表中
            	verts = out['pred_vertices'][n].detach().cpu().numpy()
            	is_right = batch['right'][n].cpu().numpy()
            	verts[:, 0] = (2 * is_right - 1) * verts[:, 0]  # 根据左右手调整顶点位置
            	cam_t = pred_cam_t_full[n]
            	all_verts.append(verts)
            	all_cam_t.append(cam_t)
            	all_right.append(is_right)

            	# 如果需要保存网格到磁盘
            	if args.save_mesh:
                	camera_translation = cam_t.copy()
                	# 将顶点转换为trimesh格式
                	tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                	# 导出网格模型到OBJ文件
                	tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))
```

上面代码提供了保存重建的手部网格数据的功能，其中渲染侧面视图将调用renderer.py中的__call__方法，部分参数说明如下：

```
#hamer/hamer/utils/renderer.py
def __call__(self,
                vertices: np.array,
                camera_translation: np.array,
                image: torch.Tensor,
                full_frame: bool = False,
                imgname: Optional[str] = None,
                side_view=False, rot_angle=90,
                mesh_base_color=(1.0, 1.0, 0.9),
                scene_bg_color=(0,0,0),
                return_rgba=False,
                ) -> np.array:
```

Render meshes on input image

​    Args:

​      vertices (np.array): Array of shape (V, 3) containing the mesh vertices.

​      camera_translation (np.array): Array of shape (3,) with the camera translation.

​      image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.

​      full_frame (bool): If True, then render on the full image.

​      imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.

后续将详细查看renderer.py文件，接下来看demo.py的最后部分

```
# 渲染正面视图
if args.full_frame and len(all_verts) > 0:
    # 设置渲染器的其他参数
    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,  # 网格的基础颜色
        scene_bg_color=(1, 1, 1),  # 场景背景颜色
        focal_length=scaled_focal_length,  # 相机的焦距
    )
    # 使用 renderer 的 render_rgba_multiple 方法渲染所有检测到的手部
    # all_verts, cam_t, img_size[n], is_right 等参数分别表示：
    # 顶点数据，相机参数，图像尺寸，左右手标识
    # **misc_args 表示将前面定义的额外参数传递给渲染函数
    cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

    # 合成图像
    # 将原始图像转换为浮点数类型，并除以 255 转换到 [0,1] 范围，然后进行 RGB 到 BGR 的颜色通道重排
    input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
    # 给图像添加 Alpha 通道（全 1），以便于后续进行图像混合
    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
    # 根据 cam_view 的 Alpha 通道进行图像混合
    # 如果 cam_view 的 Alpha 通道为 0，则使用原始图像的 RGB 值
    # 如果 cam_view 的 Alpha 通道为 1，则使用渲染图像的 RGB 值
    input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

    # 将合成图像保存到磁盘，乘以 255 转换回 [0,255] 范围
    cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255 * input_img_overlay[:, :, ::-1])

# 脚本的入口点
if __name__ == '__main__':
    main()  # 调用 main 函数执行程序
```

这段代码首先检查是否需要渲染全图（`args.full_frame`），并且是否检测到了至少一只手（`len(all_verts) > 0`）。如果这两个条件都满足，它将使用 `Renderer` 类的 `render_rgba_multiple` 方法来渲染所有手部的正面视图。然后，它将渲染结果与原始图像进行合成，创建一个叠加图像，并将这个图像保存到磁盘上，这个过程允许用户在单个图像中看到所有人的手部重建结果。

## 网络架构
```bash
#hamer/hamer/models/hamer.py
class HAMER(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])

        # Create MANO head
        self.mano_head = build_mano_head(cfg)

        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.mano_parameter_loss = ParameterLoss()

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False
```

## HInt Dataset
We have released the annotations for the HInt dataset. Please follow the instructions [here](https://github.com/ddshan/hint)

## Training
First, download the training data to `./hamer_training_data/` by running:
```
bash fetch_training_data.sh
```

Then you can start training using the following command:
```
python train.py exp_name=hamer data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```
Checkpoints and logs will be saved to `./logs/`.

## Evaluation
Download the [evaluation metadata](https://www.dropbox.com/scl/fi/7ip2vnnu355e2kqbyn1bc/hamer_evaluation_data.tar.gz?rlkey=nb4x10uc8mj2qlfq934t5mdlh) to `./hamer_evaluation_data/`. Additionally, download the FreiHAND, HO-3D, and HInt dataset images and update the corresponding paths in  `hamer/configs/datasets_eval.yaml`.

Run evaluation on multiple datasets as follows, results are stored in `results/eval_regression.csv`. 
```bash
python eval.py --dataset 'FREIHAND-VAL,HO3D-VAL,NEWDAYS-TEST-ALL,NEWDAYS-TEST-VIS,NEWDAYS-TEST-OCC,EPICK-TEST-ALL,EPICK-TEST-VIS,EPICK-TEST-OCC,EGO4D-TEST-ALL,EGO4D-TEST-VIS,EGO4D-TEST-OCC'
```

Results for HInt are stored in `results/eval_regression.csv`. For [FreiHAND](https://github.com/lmb-freiburg/freihand) and [HO-3D](https://codalab.lisn.upsaclay.fr/competitions/4318) you get as output a `.json` file that can be used for evaluation using their corresponding evaluation processes.

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [4DHumans](https://github.com/shubham-goel/4D-Humans)
- [SLAHMR](https://github.com/vye16/slahmr)
- [ProHMR](https://github.com/nkolot/ProHMR)
- [SPIN](https://github.com/nkolot/SPIN)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)
- [HMR](https://github.com/akanazawa/hmr)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [Detectron2](https://github.com/facebookresearch/detectron2)

Additionally, we thank [StabilityAI](https://stability.ai/) for a generous compute grant that enabled this work.

## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}
```
