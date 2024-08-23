# 2D Grided Gaussians

## Install
First install CUDA and PyTorch, our code is evaluated on [CUDA 11.6](https://developer.nvidia.com/cuda-11-6-2-download-archive) and [PyTorch 1.13.1+cu116](https://pytorch.org/get-started/previous-versions/#v1131). Then install the following dependencies:
```
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

Install [COLMAP](https://colmap.github.io/install.html) for calibration and undistortion. 

Install [NeuS2](https://github.com/19reborn/NeuS2/) for key frame initial point cloud generation, please clone it to `external` folder and build it. 





## Dataset Preparation

### Download Dataset

Our code is evaluated on multi-view human centric datasets including [ReRF](https://github.com/aoliao12138/ReRF_Dataset), [HiFi4G](https://github.com/moqiyinlun/HiFi4G_Dataset), and [HumanRF](https://synthesiaresearch.github.io/humanrf/#dataset) datasets. Please download the data you needed and put them in the datasets folder. Our dataset is structured as follows:
```
datasets
|---ReRF
|---HumanRF
|---HiFi4G
|   |---0923dancer3
|   |   |---0
|   |   |---1
|   |   |   |---images
|   |   |   |   |---img_0000.png
|   |   |   |   |---img_0001.png
|   |   |   |   |---...
|   |   |---...
|   |---0923dancer3_cali
|   |   |---images
```

### Calibration



### Image Undistortion

First use provided colmap calibration or your own calibration to undistort all frames. After undistortion, convert colmap datasets format to our datasets formats. Our datasets formats including a `.json` file for cameras extrinsic, intrinsic, image width and image height. The format is as follow:
```
{
    "frames": [
        {
            "file_path": "xxx/xxx.png", 
            "transform_matrix": [
                xxx
            ], 
            "K": [
                xxx
            ],
            "fl_x": xxx, 
            "fl_y": xxx,
            "cx": xxx,
            "cy": xxx,
            "w": xxx,
            "h": xxx
        }, 
        {
            ...
        }
    ], 
    "aabb_scale": xxx, 
    "white_transparent": true
}
```

Finally, the datasets should be structed as follows
```
datasets
|---ReRF
|---HumanRF
|---HiFi4G
|   |---0923dancer3
|   |   |---0
|   |   |---1
|   |   |   |---images
|   |   |   |   |---img_0000.png
|   |   |   |   |---img_0001.png
|   |   |   |   |---...
|   |   |   |---image_undistortion_white
|   |   |   |   |---images
|   |   |   |   |   |---0.png
|   |   |   |   |   |---1.png
|   |   |   |   |   |---...
|   |   |   |   |---sparse
|   |   |   |   |---stereo
|   |   |   |   |---run-colmap-geometric.sh
|   |   |   |   |---run-colmap-photometric.sh
|   |   |   |---transforms.json
```
## Train



## Compress

After getting the Gaussian point clouds, we can compress them by the following command:
```

```