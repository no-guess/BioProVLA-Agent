# BioProVLA-Agent

An Affordable, Protocol-Driven, Vision-Enhanced VLA-Enabled Embodied Multi-Agent System with Closed-Loop-Capable Reasoning for Biological Laboratory Manipulation

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)]()

[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/no-guess/BioProVLA-Agent)

[![Dataset](https://img.shields.io/badge/Dataset-Coming_Soon-orange)]()



![fig1](./assets/figures/fig1.png)



## 📽️1. Example

### 🗓️ 1.1 Single Task

| <img src="./assets/gifs/close_centrifuge.gif" alt="close_centrifuge" style="width: 300px;height: 200px" /> | <img src="./assets/gifs/open_centrifuge.gif" alt="open_centrifuge" style="width: 300px;height: 200px" /> | <img src="./assets/gifs/remove_tube_centrifuge.gif" alt="remove_tube_centrifuge" style="width: 300px;height: 200px" /> | <img src="./assets/gifs/tube_into_centrifuge.gif" alt="tube_into_centrifuge" style="width: 300px;height: 200px" /> |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./assets/gifs/discard_used_1.8ml_cryotube.gif" alt="discard_used_1.8ml_cryotube" style="width: 300px;height: 200px" /> | <img src="./assets/gifs/discard_used_15ml_cent_tube.gif" alt="discard_used_15ml_cent_tube" style="width: 300px;height: 200px" /> | <img src="./assets/gifs/insert_1.8ml_cryotube_red_rack.gif" alt="insert_1.8ml_cryotube_red_rack" style="width: 300px;height: 200px" /> | <img src="./assets/gifs/insert_15ml_centri_tube_rack.gif" alt="insert_15ml_centri_tube_rack" style="width: 300px;height: 200px" /> |
| <img src="./assets/gifs/open_water_bath_lid.gif" alt="open_water_bath_lid" style="width: 300px;height: 200px" /> | <img src="./assets/gifs/close_water_bath_lid.gif" alt="close_water_bath_lid" style="width: 300px;height: 200px" /> | <img src="./assets/gifs/place_float_water_bath.gif" alt="place_float_water_bath" style="width: 300px;height: 200px" /> | <img src="./assets/gifs/remove_float_water_bath.gif" alt="remove_float_water_bath" style="width: 300px;height: 200px" /> |

### 🗓️ 1.2 Double Task

| <img src="./assets/gifs/Unscrew_centrifuge_tube_cap.gif" alt="Unscrew_centrifuge_tube_cap" style="width: 500px;height: 220px" /> | <img src="./assets/gifs/Liquid_Waste_Pouring.gif" alt="Liquid_Waste_Pouring" style="width: 500px;height: 220px" /> | <img src="./assets/gifs/Tighten_centrifuge_cap.gif" alt="Tighten_centrifuge_cap" style="width: 500px;height: 220px" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

### 🗓️ 1.3 Composite Task

| [Loading centrifuge tube](./assets/videos/single_arm_1.mp4)  | [Unload centrifuge tube](./assets/videos/single_arm_2.mp4) | [Tidy up the desktop](./assets/videos/single_arm_3.mp4)  |
| ------------------------------------------------------------ | ---------------------------------------------------------- | -------------------------------------------------------- |
| **[Clean up waste materials](./assets/videos/single_arm_4.mp4)** | **[Loading float](./assets/videos/single_arm_5.mp4)**      | **[Unload the float](./assets/videos/single_arm_6.mp4)** |
| **[Pour Waste Liquid](./assets/videos/double_arm_1.mp4)**    |                                                            |                                                          |



## ⚙️ 2. Installation

```cmd
# Clone this repo
git clone https://github.com/no-guess/BioProVLA-Agent
cd BioProVLA-Agent/

# Create a Conda environment
conda create -n BioProVLA-Agent python=3.12 -y
conda activate BioProVLA-Agent
```



## 🤖 3. BioProVLA-Agent

### ✏️ 3.1 Prepare the environment

```cmd
# Install the direct dependencies required by BioProVLA-Agent:
pip install -r bioprovla_agent/requirements.txt

# Install the bundled LeRobot package with SmolVLA support:
conda install ffmpeg -c conda-forge
cd BioProVLA-Agent/lerobot-main
pip install -e .
pip install lerobot
pip install -e ".[smolvla]"
```

### ✏️ 3.2 **Configuration Instructions**

```cmd
# BioProVLA-Agent is configured with:
configs/bioprovla_example.json
```

### ✏️ 3.3 Start Execution

```cmd
# Run BioProVLA-Agent:
python -m bioprovla_agent.run_cli --config configs/bioprovla_example.json
```



## 🧠 4. SmolVLA Data Augmentation Training

### ✏️ 4.1 **Configuration Instructions**

```cmd
# SmolVLA data augmentation is configured through --policy.* command-line arguments.
# The main training-time augmentation options are:
--policy.enable_lighting_augmentation_training=true

# Enable lighting augmentation during training.
--policy.lighting_schedule_total_steps=30000

# Set the total number of training steps for the lighting augmentation schedule.
--policy.enable_lighting_visualization_save=false

# Whether to save original/enhanced image pairs during training.
--policy.lighting_visualization_dir=lighting_visualizations
```

### ✏️ 4.2 Start Train

```cmd
# Enable fixed lighting scenario processing during inference.
lerobot-train 
  --dataset.repo_id=/lerobot/insert_1.8ml_cryotube_red_rack/demo \
  --output_dir=outputs/train/random_smolvla_insert_1.8ml_cryotube_red_rack \
  --job_name=smolvla_insert_1.8ml_cryotube_red_rack_so101_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --policy.path=lerobot/smolvla_base  \
  --batch_size=32 \
  --steps=30000 \
  --policy.gradient_accumulation_steps=2 \
  --policy.enable_lighting_augmentation_training=true \
  --policy.lighting_schedule_total_steps=30000 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.handeye": "observation.images.camera2"}'
  
# Test
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=your save dir \
  --dataset.single_task="Place the 1.8 mL cryotube into the red cryotube rack" \
  --policy.path=your train model dir
  --dataset.push_to_hub=False \
  --robot.calibration_dir=your calibration dir
```

## 🙏 Acknowledgements

We sincerely thank the LeRobot team for open-sourcing their official codebase, which provides an important foundation and reference for the development of this project.
