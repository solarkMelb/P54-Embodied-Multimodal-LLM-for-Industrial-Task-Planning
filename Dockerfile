# # Use Python 3.11 on Ubuntu — PyBullet works on Linux, not macOS
# FROM python:3.11-slim-bullseye

# # Install system dependencies for PyBullet, OpenCV, and C/C++ compilation
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     g++ \
#     python3-dev \
#     libgl1-mesa-glx \
#     libgl1-mesa-dri \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     x11-apps \
#     xvfb \
#     ffmpeg \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /app

# # Copy requirements first (layer caching)
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Install PyBullet explicitly (needs Linux)
# RUN pip install --no-cache-dir pybullet

# # Copy entire project
# COPY . .


# COPY entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh
# ENTRYPOINT ["/entrypoint.sh"]


# RUN pip install --no-cache-dir httpx


# COPY wait_for_ollama.sh /wait_for_ollama.sh
# RUN chmod +x /wait_for_ollama.sh

# # Use ROS2 Humble base image (Ubuntu 22.04)
# FROM osrf/ros:humble-desktop

# # Install Python and pip
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     python3-colcon-common-extensions \
#     ros-humble-moveit \
#     ros-humble-control-msgs \
#     libgl1-mesa-glx \
#     xvfb \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# WORKDIR /app
# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip3 install --no-cache-dir pybullet langchain-ollama

# COPY . .

# # Source ROS2 in every shell
# RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# # Default command
# CMD ["python", "main.py", "--interactive"]


FROM python:3.11-slim

# libgl1/libglib2.0-0 and the libx* stubs are required by opencv-python —
# without them `import cv2` fails with "libGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "main.py", "--interactive"]