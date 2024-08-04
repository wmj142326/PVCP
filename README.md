# PVCP
Pedestrian-Centric 3D Pre-collision Pose and Shape Estimation from Dashcam Perspective

## Dependencies
![python = 3.7.16](https://img.shields.io/badge/python-3.7.16-green)
![torch = 1.11.0+cu113](https://img.shields.io/badge/torch-1.11.0%2Bcu113-yellowgreen)

Some of our code and dependencies was adapted from [MotionBERT](https://github.com/Walter0807/MotionBERT).

## PVCP Dataset
![PVCP](image/PVCP_dataset.png)
We have provided a special Tool for SMPL annotation: [SMPL_Tools](https://anonymous.4open.science/r/SMPL_Tools-0C7A).

Download the PVCP Dataset: [PVCP_Dataset](https://github.com/).
Directory structure: (Coming soon...)

```shell script
PVCP
├── annotation
│   ├── dataset_2d.json
│   ├── dataset_3d.json
│   ├── mesh_det_pvcp_train_gt2d_test_det2d.pkl
│   ├── mesh_det_pvcp_train_release.pkl
│   └── train_test_seq_id_list.json
├── frame
│   └── image2frame.py
├── image
│   ├── S000_1280x720_F000000_T000000.png
│   ├── S000_1280x720_F000001_T000001.png
│   ├── S000_1280x720_F000002_T000002.png
│   ├── ...
│   └── S208_1584x660_F000207_T042510.png
├── video
│   ├── S000_1280x720.mp4
│   ├── S001_1280x720.mp4
│   ├── S002_1280x720.mp4
│   ├── ...
│   └── S208_1584x660.mp4
└── vis_2dkpt_ann.mp4
```
For the `frame` folder, run `image2frame.py`. The folder structure is as follows:
```shell script

├── frame
   ├── frame_000000.png
   ├── frame_000001.png
   ├── frame_000002.png
   ├── ...
   └── frame_042510.png
```
## PPSENet Framework
![PPSENet](image/framework_pipline.png)

## Train: 
Finetune from a pretrained model with PVCP
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mesh_pvcp.py \
--config configs/mesh/MB_ft_pvcp.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint mycheckpoint/mesh
```

## Evaluate
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mesh_pvcp.py \
--config configs/mesh/MB_ft_pvcp.yaml \
--evaluate checkpoint/mesh/best_epoch.bin 
```

## Test and Demo
```shell script
python infer_wild_mesh_list.py --out_path myoutput/qualitation/mesh_pvcp_test_from_2ddet
```

## Visual

<table>
  <tr>
    <td><img src="image/seq_1.png" alt="PNG 1" width="380"/></td>
    <td><img src="image/seq_2.png" alt="PNG 2" width="380"/></td>
  </tr>
  <tr>
    <td><img src="image/50.gif" alt="GIF 1" width="380"/></td>
    <td><img src="image/54.gif" alt="GIF 2" width="380"/></td>
  </tr>
  <tr>
    <td><img src="image/63.gif" alt="GIF 3" width="380"/></td>
    <td><img src="image/74.gif" alt="GIF 4" width="380"/></td>
  </tr>
  <tr>
    <td><img src="image/72.gif" alt="GIF 5" width="380"/></td>
    <td><img src="image/87.gif" alt="GIF 6" width="380"/></td>
  </tr>
</table>


This readme file is going to be further updated.