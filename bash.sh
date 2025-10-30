#!/usr/bin/env bash
set -euo pipefail

PYTHON=python3
VENV_DIR=.venv

echo "Creating virtualenv ${VENV_DIR}..."
${PYTHON} -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

pip install --upgrade pip setuptools wheel
echo "Installing pip packages from requirements.txt..."
pip install -r requirements.txt || {
  echo "pip install failed; check for missing system packages. See apt-get suggestions below."
  exit 1
}

echo "Installing CPU PyTorch wheel (adjust for CUDA/Jetson as needed)..."
pip install --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || true

echo
echo "Done. Activate the environment: source ${VENV_DIR}/bin/activate"
echo
echo "Apt packages you may need on Ubuntu:"
echo "  sudo apt-get update && sudo apt-get install -y build-essential cmake libssl-dev libffi-dev python3-dev ffmpeg libglib2.0-0"
echo
echo "ROS2: install rclpy and message packages via apt on Ubuntu/ROS images (do not pip install rclpy)."
echo "Jetson: install JetPack and Jetson-compatible PyTorch wheel per NVIDIA docs."

pip install --upgrade truss 'pydantic>=2.0.0'

$ truss push

sudo apt install ros-<ros2-distro>-rtabmap-ros

ros2 launch rtabmap_ros rtabmap.launch.py \
    rgb_topic:=/camera/color/image_raw \
    depth_topic:=/camera/depth/image_raw \
    scan_topic:=/lidar

$ truss init hello-world
? 📦 Name this model: HelloWorld
Truss HelloWorld was created in ~/hello-world


curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
pyenv install 3.11.0
ENV_NAME="truss_env"
pyenv virtualenv 3.11.0 $ENV_NAME
pyenv activate $ENV_NAME
pip install --upgrade truss 'pydantic>=2.0.0'

Neurobot/
├── ros2/
│   ├── launch/
│   │   └── neurobot_slam.launch.py       # Launch SLAM + sensors
│   ├── nodes/
│   │   ├── lidar_node.py                 # LiDAR publisher
│   │   ├── imu_node.py                   # IMU publisher
│   │   ├── camera_node.py                # Camera publisher
│   │   └── motor_node.py                 # Subscribes commands, controls motors
│   ├── maps/
│   │   └── saved_maps/                   # Store generated 3D maps
│   └── swarm_node.py                      # MQTT/ROS2 topic for swarm coordination


git clone https://github.com/Web4application/Brain.git
cd Brain

git clone https://github.com/Web4application/EDQ-AI.git
cd EDQ-AI

git clone https://github.com/Web4application/SERAI.git
cd SERAI


+-------------------+
|   Arduino Board   |
|-------------------|
| Sensors: Temp,   |
| Motion, Light,   |
| Distance, etc.   |
|                   |
| Actuators: Motors, LEDs, Relays |
+-------------------+
          │
          │  Sensor Data / Control Signals
          ▼
+-------------------+
|     EDQ AI        |
|-------------------|
| Data Processing   |
| Filtering /       |
| Aggregation       |
+-------------------+
          │
          │  Structured & Clean Data
          ▼
+-------------------+
|      SERAI AI     |
|-------------------|
| Reasoning Engine  |
| Simulation        |
| Predictive Models |
| Decision Making   |
+-------------------+
          │
          │  Commands / Actions
          ▼
+-------------------+
|   Arduino Board   |
| (Execution Layer) |
| Motors, LEDs,     |
| Relays, etc.      |
+-------------------+
          │
          ▼
      Real World

  mkdir -p Brain/config
mkdir -p Brain/brain/sensors
mkdir -p Brain/brain/models
mkdir -p Brain/brain/swarm
mkdir -p Brain/brain/actuators
mkdir -p Brain/brain/utils
mkdir -p Brain/examples
mkdir -p Brain/tests
mkdir -p Brain/docker
mkdir -p Brain/scripts    
