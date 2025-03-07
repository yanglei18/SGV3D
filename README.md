<p align="center">

  <h1 align="center">SGV3D:Towards Scenario Generalization for Vision-based Roadside 3D Object Detection
</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=EUnI2nMAAAAJ&hl=zh-CN"><strong>Lei Yang</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=0Q7pN4cAAAAJ&hl=zh-CN&oi=sra"><strong>Xinyu Zhang</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong>Jun Li</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=kLTnwAsAAAAJ&hl=zh-CN&oi=sra"><strong>Li Wang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=QTYSI1gAAAAJ&hl=zh-CN"><strong>Chuang Zhang</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong>Li Ju</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong>Zhiwei Li</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong>Yang Shen</strong></a>

  </p>

<div align="center">
  <img src="./assets/framework.jpg" alt="Logo" width="90%">
</div>

<p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
     <a href='https://hub.docker.com/repository/docker/yanglei2024/op-bevheight/general'><img src='https://img.shields.io/badge/Docker-9cf.svg?logo=Docker' alt='Docker'></a>
    <br></br>
    <a href="https://arxiv.org/abs/2401.16110">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
  </p>
</p>

**SGV3D** is an innovative scenario generalization framework for vision-based roadside 3D object detection. SGV3D, **with only a minimal increase in latency**, significantly outperforms other leading detectors by substantial margins of **+42.57%**, **+5.87%**, and **+14.89%** for three categories in DARI-V2X-I heterologous settings.

# News

- [2025/03/07] Both arXiv and codebase are released!

<br/>

# Getting Started

- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/run_and_eval.md)

# Acknowledgment
This project is not possible without the following codebases.
* [BEVHeight](https://github.com/ADLab-AutoDrive/BEVHeight)
* [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
* [SAM](https://github.com/facebookresearch/segment-anything)
* [pypcd](https://github.com/dimatura/pypcd)

# Citation
If you use BEVHeight in your research, please cite our work by using the following BibTeX entry:
```
@article{yang2024sgv3d,
  title={SGV3D: Towards scenario generalization for vision-based roadside 3D object detection},
  author={Yang, Lei and Zhang, Xinyu and Li, Jun and Wang, Li and Zhang, Chuang and Ju, Li and Li, Zhiwei and Shen, Yang},
  journal={arXiv preprint arXiv:2401.16110},
  year={2024}
}

```
