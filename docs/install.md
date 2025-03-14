# Step-by-step installation instructions

## Recommended docker image
```shell
docker pull yanglei2024/op-bevheight:base
```

## Installation
**a.** Install [pytorch](https://pytorch.org/)(v1.9.0).

**b.** Install mmcv-full==1.4.0  mmdet==2.19.0 mmdet3d==0.18.1.

**c.** Install pypcd
```
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install
```

**d.** Install SAM.

```
git clone https://github.com/facebookresearch/segment-anything
cd segment-anything; pip install -e .
```

**e.** Install requirements.
```shell
pip install -r requirements.txt
```
**f.** Install SGV3D (gpu required).
```shell
python setup.py develop
```
