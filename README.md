# Pedestrian-Centric 3D Pre-collision Pose and Shape Estimation from Dashcam Perspective


## Dependencies
![python = 3.7.16](https://img.shields.io/badge/python-3.7.16-green)
![torch = 1.11.0+cu113](https://img.shields.io/badge/torch-1.11.0%2Bcu113-yellowgreen)

Some of our code and dependencies was adapted from [MotionBERT](https://github.com/Walter0807/MotionBERT). 


## PVCP Dataset
![PVCP](image/PVCP_dataset.png)
We have provided a special Tool for SMPL annotation: [SMPL_Tools](https://anonymous.4open.science/r/SMPL_Tools-0C7A).

Download the PVCP Dataset: [PVCP_Dataset (Coming soon...)](https://github.com/).
Directory structure: 

```commandline
PVCP
├── annotation
│   ├── dataset_pose.json
│   ├── dataset_mesh.json
│   ├── mb_input_det_pose.json
│   ├── train_test_seq_id_list.json
│   ├── mesh_det_pvcp_train_release.pkl
│   └── mesh_det_pvcp_train_gt2d_test_det2d.pkl
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

### Project Directory Structure
```commandline
PVCP
├── checkpoint
├── configs
│   ├── mesh
│   └── pretrain
├── data
│   ├── mesh
│   └── pvcp
├── lib
│   ├── data
│   ├── model
│   └── utils
├── params
│   ├── d2c_params.pkl
│   └── synthetic_noise.pth
├── tools
│   ├── compress_amass.py
│   ├── convert_amass.py
│   ├── convert_h36m.py
│   ├── convert_insta.py
│   └── preprocess_amass.py
├── LICENSE
├── README_MotionBERT.md
├── requirements.txt
├── train_mesh_pvcp.py
└── infer_wild_mesh_list.py

```


### Data
1. Download the other datasets [here](https://1drv.ms/f/s!AvAdh0LSjEOlfy-hqlHxdVMZxWM) and put them to  `data/mesh/`. We use Human3.6M, COCO, and PW3D for training and testing. Descriptions of the joint regressors could be found in [SPIN](https://github.com/nkolot/SPIN/tree/master/data).
2. Download the SMPL model(`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`) from [SMPLify](https://smplify.is.tue.mpg.de/), put it  to `data/mesh/`, and rename it as `SMPL_NEUTRAL.pkl`
3. Download the PVCP dataset [here (Coming soon...)](https://github.com/) and put them to  `data/pvcp/`. mv `mesh_det_pvcp_train_release.pkl` and `mesh_det_pvcp_train_gt2d_test_det2d.pkl` to `data/mesh/`.
    
4. You can also download our `data (Coming soon...)` and `checkpoint (Coming soon...)` folders directly. Final, `data` directory structure as follows:
    ```commandline
    ├── data
        ├── mesh
        │   ├── J_regressor_extra.npy
        │   ├── J_regressor_h36m_correct.npy
        │   ├── mesh_det_coco.pkl
        │   ├── mesh_det_h36m.pkl
        │   ├── mesh_det_pvcp_train_gt2d_test_det2d.pkl
        │   ├── mesh_det_pvcp_train_release.pkl
        │   ├── mesh_det_pw3d.pkl
        │   ├── mesh_hybrik.zip
        │   ├── smpl_mean_params.npz
        │   └── SMPL_NEUTRAL.pkl
        └── pvcp
            ├── annotation
            │   ├── mb_input_det_pose.json
            │   └── train_test_seq_id_list.json
            ├── frame
            ├── image
            └── video
    ```


### Train: 
Finetune from a pretrained model with PVCP
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mesh_pvcp.py \
--config configs/mesh/MB_ft_pvcp.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/mesh/ft_pvcp_iter3_class0.1_gt_release
```

### Evaluate
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mesh_pvcp.py \
--config configs/mesh/MB_ft_pvcp.yaml \
--evaluate checkpoint/mesh/ft_pvcp_iter3_class0.1_gt_release/best_epoch.bin 
```

### Test and Demo
```shell script
python infer_wild_mesh_list.py --out_path output/
```


### Visual

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


Our paper is undergoing a double-blind review and all links will be published after publication.