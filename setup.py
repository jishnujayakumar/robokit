#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

import configparser
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import setuptools
from setuptools.command.install import install


def _has_cuda_toolkit():
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        return False
    return os.path.exists(cuda_home)

class FileFetch(install):
    """
    Custom setuptools command to fetch required files from external sources.
    """
    def run(self):
        """
        Execute the command to fetch required files.
        """
        if os.environ.get("ROBOTKIT_ENABLE_FILEFETCH") != "1":
            logging.info(
                "ROBOTKIT_ENABLE_FILEFETCH not set to 1; skipping heavy FileFetch steps."
            )
            install.run(self)
            return
        install.run(self)

        robokit_root_dir = os.getcwd()

        requirements_file = Path(robokit_root_dir) / "requirements.txt"
        if requirements_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)

        # Install the dependency from the Git repository
        git_packages = [
            'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33',
            'git+https://github.com/IDEA-Research/GroundingDINO.git@2b62f419c292ca9c518daae55512fabc3fead4a4',
            'git+https://github.com/ChaoningZhang/MobileSAM@c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed',
        ]
        if _has_cuda_toolkit():
            git_packages.append('git+https://github.com/mhamilton723/FeatUp@c04e4c19945ce3e98a5488be948c7cc6')
        else:
            print("WARNING: CUDA toolkit not detected. Skipping FeatUp install.")
        subprocess.run(["pip", "install", "-U", *git_packages], check=True)


        # Step DHYOLO.1: Clone the DH-YOLO repository
        try:
            subprocess.run(["git", "clone", "https://github.com/IRVLUTD/iTeach"], check=True)
        except:
            pass

        # Step DHYOLO.2: Copy the required folder
        subprocess.run(["cp", "-r", "iTeach/toolkit/iteach_toolkit", "robokit"], check=True)

        # Step DHYOLO.3: Copy the required folder
        subprocess.run(["rm", "-rf", "iTeach"], check=True)



        # Step SAMv2.1: Clone the repository
        samv2_dir = os.path.join(robokit_root_dir, "robokit", "sam2")
        os.makedirs(samv2_dir, exist_ok=True)
        try:
            subprocess.run(["git", "clone", "https://github.com/facebookresearch/sam2", samv2_dir], check=True)
        except:
            pass

        # Step SAMv2.2: cd to samv2 and checkout the desired commit branch
        os.chdir(samv2_dir)
        subprocess.run(["git", "checkout", "--branch", "c2ec8e14a185632b0a5d8b161928ceb50197eddc"])

        # Step SAMv2.3: Use sed to comment out line 171 (to get rid of py>=3.10)
        subprocess.run(["sed", "-i", "171s/^/#/", "setup.py"], check=True)

        # Step SAMv2.4: Install samv2
        subprocess.run(["python", "setup.py", "install"], check=True)

        # Step SAMv2.5: move to robokit root directory
        os.chdir(robokit_root_dir)        

        # subprocess.run([
        #     "conda", "install", "-y", "pytorch", "torchvision", "torchaudio", "pytorch-cuda=11.7", "-c", "pytorch", "-c", "nvidia"
        # ])

        subprocess.call


        # Download GroundingDINO checkpoint
        self.download_pytorch_checkpoint(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            os.path.join(os.getcwd(), "ckpts", "gdino"),
            "gdino.pth"
        )
        
        ##############################################################################################################

        # Download SAM checkpoint
        # self.download_pytorch_checkpoint(
        #     "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        #     os.path.join(os.getcwd(), "ckpts", "sam"),
        #     "vit_h.pth"
        # )

        # Download SAM checkpoint
        self.download_pytorch_checkpoint(
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
            os.path.join(os.getcwd(), "ckpts", "mobilesam"),
            "vit_t.pth"
        )
        
        ##############################################################################################################

        # Download DHYOLO checkpoints
        dhyolo_checkpoints = [
            "dh-yolo-v1-pb-ddf-524.pt",
            "dh-yolo-exp27-pb-1008.pt",
            "dh-yolo-exp-31-pl-1532.pt",
            "dh-yolo-exp-31-pb-1532.pt"
        ]

        for ckpt in dhyolo_checkpoints:
            self.download_pytorch_checkpoint(
                f"https://huggingface.co/spaces/IRVLUTD/DH-YOLO/resolve/main/pretrained_ckpts/{ckpt}",
                os.path.join(os.getcwd(), "ckpts", "dhyolo"),
                ckpt
            )
        
        ##############################################################################################################
        
        # Download SAM2 checkpoint
        self.download_pytorch_checkpoint(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            os.path.join(os.getcwd(), "ckpts", "samv2"),
            "sam2.1_hiera_large.pth"
        )

        # Download SAM2 checkpoint yaml (exploiting the download ckpt method's download nature)
        self.download_pytorch_checkpoint(
            "https://raw.githubusercontent.com/facebookresearch/sam2/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            os.path.join(os.getcwd(), "ckpts", "samv2"),
            "sam2.1_hiera_l.yaml"
        )
        
        ##############################################################################################################


    def download_pytorch_checkpoint(self, pth_url: str, save_path: str, renamed_file: str):
        """
        Download a PyTorch checkpoint from the given URL and save it to the specified path.

        Parameters:
        - pth_url (str): The URL of the PyTorch checkpoint file.
        - save_path (str): The path where the checkpoint will be saved.
        - renamed_file (str, optional): The desired name for the downloaded file.

        Raises:
        - FileNotFoundError: If the file cannot be downloaded or saved.
        - Exception: If an unexpected error occurs during the download process.
        """
        try:
            import requests
            from tqdm import tqdm
            file_path = os.path.join(save_path, renamed_file)

            # Check if the file already exists
            if os.path.exists(file_path):
                logging.info(f"{file_path} already exists! Skipping download")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)

            # Log download attempt
            logging.info("Attempting to download PyTorch checkpoint from: %s", pth_url)


            response = requests.get(pth_url, stream=True)
            response.raise_for_status()  # Raise an HTTPError for bad responses

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            # Save the checkpoint to the specified path
            with open(file_path, 'wb') as file:
                for data in response.iter_content(chunk_size=block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            progress_bar.close()

            logging.info("Checkpoint downloaded and saved to: %s", file_path)

        except FileNotFoundError as e:
            logging.error("Error: Checkpoint file not found: %s", e)
            raise e


# Read requirements from requirements.txt if present; otherwise fall back
DEFAULT_REQUIREMENTS = [
    "chardet",
    "requests",
    "tqdm",
    "ftfy",
    "regex",
    "torch",
    "torchvision",
    "absl-py",
    "open3d",
    "scikit-image",
    "numpy==1.26.4",
    "supervision==0.18.0",
    "rospkg",
    "transforms3d",
]
requirements_path = Path(__file__).with_name("requirements.txt")
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = f.read().splitlines()
else:
    logging.warning("requirements.txt not found; using default dependency list.")
    requirements = DEFAULT_REQUIREMENTS.copy()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

extras = {
    "gdino": [
        "clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33",
        "GroundingDINO @ git+https://github.com/IDEA-Research/GroundingDINO.git@2b62f419c292ca9c518daae55512fabc3fead4a4",
    ],
    "sam": [
        "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM@c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed",
    ],
    "sam2": [
        "hydra-core>=1.3.2",
    ],
    "featup": [
        "FeatUp @ git+https://github.com/mhamilton723/FeatUp@c04e4c19945ce3e98a5488be948c7cc6",
    ],
    "dhyolo": [
        "ultralytics>=8.0.0",
    ],
    "depthany": [
        "transformers>=4.34.1",
    ],
}
if not _has_cuda_toolkit():
    print("WARNING: CUDA toolkit not detected. Skipping automatic FeatUp install.")
    extras["featup"] = []

extras["all"] = sorted({dep for deps in extras.values() for dep in deps})


def _extras_from_config(config_path: Path) -> list[str]:
    """Load extras to install by default from an INI-style config file."""
    if not config_path.exists():
        return []
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if not parser.has_option("extras", "include"):
        return []
    values = parser.get("extras", "include")
    return [value.strip() for value in values.split(",") if value.strip()]


config_extras = _extras_from_config(Path(__file__).with_name("robokit.install.cfg"))
if config_extras:
    known = []
    unknown = []
    for extra_name in config_extras:
        if extra_name in extras:
            known.append(extra_name)
        else:
            unknown.append(extra_name)
    if unknown:
        logging.warning("Unknown extras declared in robokit.install.cfg: %s", ", ".join(unknown))
    if known:
        logging.info("Auto-including extras from config: %s", ", ".join(known))
        for name in known:
            requirements.extend(extras[name])

setuptools.setup(
    name="RoboKit",
    version="0.0.1",
    author="Jishnu P",
    author_email="jishnu.p@utdallas.edu",
    description="A toolkit for robotic tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IRVLUTD/RoboKit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.0",
    install_requires=requirements,
    cmdclass={
        'install': FileFetch,
    },
    extras_require=extras,
    include_package_data=True,
    package_data={'robokit': ["cfg/gdino/*.py"]}
)
