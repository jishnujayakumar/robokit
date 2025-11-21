# 🤖 RoboKit
A toolkit for robotic tasks

## 🚀 Projects Using RoboKit
Chronologically listed (latest first):
- Perception: [MRVG](https://irvlutd.github.io/MultiGrounding/)
- Mobile Manipulation: [HRT1](https://irvlutd.github.io/HRT1/)
- Interactive Robot Teaching: [iTeach](https://irvlutd.github.io/iTeach/)
- Robot Exploration and Navigation: [AutoX-SemMap](https://irvlutd.github.io/SemanticMapping/) 
- Perception Research: [NIDS-Net](https://irvlutd.github.io/NIDSNet)
- Grasp Trajectory Optimization: [GTO](https://irvlutd.github.io/GraspTrajOpt/)

## ✨ Features
- **Docker Support**
  - Base image with ROS Noetic + CUDA 11.8 + Ubuntu 20.04 + Gazebo 11  
    → [`Dockerfile`](docker/Dockerfile-ub20.04-ros-noetic-cuda11.8-gazebo)
  - Refer to BundleSDF's [Docker setup](https://github.com/NVlabs/BundleSDF?tab=readme-ov-file#dockerenvironment-setup)
  - Quickstart script: [`run_container.sh`](docker/run_container.sh)

- **Zero-Shot Capabilities**
  - 🔍 CLIP-based classification  
  - 🎯 Text-to-BBox: GroundingDINO  
  - 🧼 BBox-to-Mask: Segment Anything (MobileSAM)  
  - 📏 Image-to-Depth: Depth Anything  
  - 🔼 Feature Upsampling: FeatUp  
  - 🚪 DoorHandle Detection: iTeach–DHYOLO ([demo](https://huggingface.co/spaces/IRVLUTD/DH-YOLO))  
  - 📽️ Mask Propagation for Videos: SegmentAnythingV2 (SAMv2)
    - Input: `jpg` or `mp4`
    - Supports:
      - Point/BBox prompts across video frames
      - Multi-object point collection
    - Tip: Use jpgs for frame-wise prediction; skip conversion for single images
    - Note that SAMv2 only supports mp4 or jpg files as of 11/06/2024
    - If you have an mp4 file then extract individual frames as jpg and store in a directory
    - For single image mask predictions, no need to convert to jpg.

## ⚙️ Getting Started

### 🧰 Prerequisites
- Python 3.7 or higher (tested 3.9.18)
- torch (tested 2.0)
- torchvision
- pytorch-cuda=11.8 (tested)
- [SAMv2 requires py>=3.10.0](https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/setup.py#L171) (here the installation has been tweaked to remove this constraint)

### 🛠️ Installation
RoboKit relies on upstream git repositories (GroundingDINO, MobileSAM, FeatUp, etc.), so the supported workflow is to clone the repository and install from source.

```sh
git clone https://github.com/IRVLUTD/robokit.git
cd robokit
pip install -e .
# or configure extras via robokit.install.cfg and rerun `pip install -e .`
# or selectively: pip install -e '.[gdino,sam2]'
```

To use the config-based workflow, edit `robokit.install.cfg` and set the comma-separated extras you want:

```
[extras]
include = gdino, sam2
```

Running `pip install -e .` now installs the listed extras automatically—handy for team-wide setups without long pip commands.

Available extras:

| Extra      | Includes                                                                          |
|------------|-----------------------------------------------------------------------------------|
| `gdino`    | GroundingDINO + CLIP                                                              |
| `sam`      | MobileSAM                                                                         |
| `sam2`     | SAM v2 + Hydra                                                                    |
| `depthany` | Depth Anything (Transformers)                                                     |
| `dhyolo`   | Ultralytics + DHYOLO toolkit                                                      |
| `featup`   | FeatUp (requires CUDA toolkit and `CUDA_HOME`)                                    |
| `all`      | Installs every extra supported on your environment                                |

**Fetching checkpoints**

The default install skips large asset downloads so it works in offline/CI environments. When you need RoboKit to fetch and stage pretrained checkpoints (GDINO, MobileSAM, DHYOLO, SAMv2), rerun the install with:

```sh
ROBOTKIT_ENABLE_FILEFETCH=1 pip install -e .
```

> Checkpoints (GDINO, MobileSAM, DHYOLO, SAMv2) are downloaded when `ROBOTKIT_ENABLE_FILEFETCH=1` is set before installation (e.g., `ROBOTKIT_ENABLE_FILEFETCH=1 python setup.py install`). Without it, installations skip the heavy downloads, and you can trigger them manually later when needed. If a CUDA toolkit is not detected, the FeatUp install is skipped with a warning.

🧩 Known Installation Issues
- Check GroundingDINO [installation](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) for the following error
```sh
NameError: name '_C' is not defined
```
- For SAMv2, `ModuleNotFoundError: No module named 'omegaconf.vendor'`
```sh
pip install --upgrade --force-reinstall hydra-core
```

🧪 Usage
- Note: All test scripts are located in the [`test`](test) directory. Place the respective test scripts in the root directory to run.
- SAM: [`test_sam.py`](test/test_sam.py)
- GroundingDINO + SAM: [`test_gdino_sam.py`](test/test_gdino_sam.py)
- GroundingDINO + SAM + CLIP: [`test_gdino_sam_clip.py`](test/test_gdino_sam_clip.py)
- Depth Anything: [`test_depth_anything.py`](test/test_depth_anything.py)
- FeatUp: [`test_featup.py`](test/test_featup.py)
- iTeach-DHYOLO: [`test_dhyolo.py`](test/test_dhyolo.py)
- SAMv2: 
  - [`collect_point_prompts.py`](test/collect_point_prompts.py)
  - [`test_samv2_1_bbox_prompt.py`](test/test_samv2_1_bbox_prompt.py)
  - [`test_samv2_point_prompts.py`](test/test_samv2_point_prompts.py)
  - [`test_gdino_sam2_img.py`](test/test_gdino_sam2_img.py)
- Test Datasets: [`test_dataset.py`](test/test_dataset.py)
  - `python test_dataset.py --gpu 0 --dataset <ocid_object_test/osd_object_test>`

## 🛣️ Roadmap
Planned improvements:
- Config-based pretrained checkpoint switching
- ✨ More features coming soon...


## 🙏 Acknowledgments

This project is based on the following repositories (license check mandatory):
- [CLIP](https://github.com/openai/CLIP)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [DepthAnything](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything#transformers.DepthAnythingForDepthEstimation)
- [FeatUp](https://github.com/mhamilton723/FeatUp)
- [iTeach](https://irvlutd.github.io/iTeach/)-[DHYOLO](https://huggingface.co/spaces/IRVLUTD/DH-YOLO)
- [SAMv2](https://github.com/facebookresearch/sam2)


Special thanks to Dr. [Yu Xiang](https://yuxng.github.io/), [Sai Haneesh Allu](https://saihaneeshallu.github.io/), and [Itay Kadosh](https://scholar.google.com/citations?user=1ZLE5jsAAAAJ&hl=en) for their early feedback.

## 📜 License
This project is licensed under the MIT License. However, before using this tool please check the respective works for specific licenses.
