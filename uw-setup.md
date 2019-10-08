# Setting Up GenRe On DGX

We will mostly be mirroring instructions from **Environment Setup** from the [primary readme](https://github.com/JoshuaBrockschmidt/GenRe-ShapeHD/tree/uw). There are a few minor caveats to get things working on DGX.

## Environment Setup

### Installing Anaconda

Make sure you have Anaconda installed. If Anaconda is not installed on your system, seek other tutorials for installing it locally. You will need to make sure your Anaconda bin is in your PATH. If you installed Anaconda locally, add the following to your `.bashrc`:
```
PATH="$PATH:/home/yourusername/anaconda3/bin"
```

### Installing GenRe/ShapeHD Components

1. Ensure Python 3.6 or higher and CUDA 9.0 is installed.

1. Clone our custom version of the repo with
   ```
   # cd to the directory you want to work in
   git clone -b uw https://github.com/joshuabrockschmidt/GenRe-ShapeHD.git
   cd GenRe-ShapeHD
   ```

1. Create a conda environment named `shaperecon` with necessary dependencies specified in `environment.yml`. In order to make sure trimesh is installed correctly, please run `install_trimesh.sh` after setting up the conda environment.
   ```
   conda env create -f environment.yml
   ./install_trimesh.sh
   ```
   Note that most scripts activate the `shaperecon` Anaconda environment for you. To activate the environment manually at any time, use
   ```
   source activate shaperecon
   ```

1. The instructions below assume you have activated this environment and built the cuda extension with
   ```
   source activate shaperecon
   ./build_toolbox.sh
   ```
 We made some slight modifications to this script's child scripts for CUDA compatibility with NVIDIA Tesla P100-SXM2 GPUs.

Note that due to the deprecation of cffi from pytorch 1.0 and on, this only works for pytorch 0.4.1.

## Models and Data

### Downloading Trained Models

To download the trained GenRe and ShapeHD models (1 GB in total), `cd` into your copy of the repo and run
```
wget http://genre.csail.mit.edu/downloads/genre_shapehd_models.tar -P downloads/models/
tar -xvf downloads/models/genre_shapehd_models.tar -C downloads/models/
```

* GenRe: `depth_pred_with_inpaint.pt` and `full_model.pt`
* ShapeHD: `marrnet1_with_minmax.pt` and `shapehd.pt`

The directory location of these downloaded models is important.

### Downloading ShapeNet

This repo comes with a few Pix3D images and ShapeNet renderings, located in `downloads/data/test`, for testing purposes.

For training or a more extensive evaluation, a dataset is made available with RGB and 2.5D sketch renderings, paired with their corresponding 3D shapes, for ShapeNet cars, chairs, and airplanes, with each object captured in 20 random views. Note that this `.tar` is 143 GB and contains data not included in the original ShapeNet dataset. Expect it to take roughly a day to download,
```
wget http://genre.csail.mit.edu/downloads/shapenet_cars_chairs_planes_20views.tar -P downloads/data/
mkdir downloads/data/shapenet/
tar -xvf downloads/data/shapenet_cars_chairs_planes_20views.tar -C downloads/data/shapenet/
```
If you are using DGX, this dataset is already downloaded at `/raid/jcbrock/shapenet`. You can simply create a symbolic link to the dataset,
```
ln -s /raid/jcbrock/shapenet downloads/data/shapenet
```

### Testing GenRe

It is necessary to run the following scripts from within the base directory of the repository.

To test GenRe against a small testing set (included with the repository) run
```
./scripts/test_genre.sh <GPUS>
```
where `<GPUS>` specifies which GPUs, e.g. `1,2,3,4` or `-1` to use CPU only.

To evaluate GenRe against the validation split of the ShapeNet dataset run
```
./scripts/eval_genre.sh <GPUS>
```
This will output the losses for each sample to `./output/eval_genre_full_model/loss_data.csv`

