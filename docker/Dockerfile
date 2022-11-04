FROM nvidia/cudagl:11.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

#####################################################
# Install common apt packages
#####################################################
RUN apt-get update && apt-get install -y \
	### utility
	locales \
	xterm \
	dbus-x11 \
	terminator \
	sudo \
	### tools
	unzip \
	lsb-release \
	curl \
	ffmpeg \
	net-tools \
	software-properties-common \
	subversion \
	libssl-dev \
	xvfb \
	### Development tools
	build-essential \
	htop \
	git \
	vim \
	gedit \
	gdb \
	valgrind \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*


#####################################################
# Set locale & time zone
#####################################################
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ=Asia/Tokyo


#####################################################
# cmake 3.15.5
#####################################################
RUN git clone https://gitlab.kitware.com/cmake/cmake.git && \
	cd cmake && \
	git checkout tags/v3.16.3 && \
	./bootstrap --parallel=8 && \
	make -j8 && \
	make install && \
	cd .. && rm -rf cmake


#####################################################
# Python 3.8
#####################################################
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
  python3.8 \
  python3-pip \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*
RUN echo "alias python='python3'" >> /root/.bashrc


#####################################################
# Install common pip packages
#####################################################
COPY pip/requirements.txt requirements.txt
RUN pip install -r requirements.txt


#####################################################
# ROS
#####################################################
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update && apt-get install -y \
	libfcl* \
	libglew-dev \
	ros-noetic-desktop-full \
	ros-noetic-joy \
	ros-noetic-gazebo* \
	ros-noetic-moveit* \
	ros-noetic-image-view* \
	ros-noetic-cv-camera* \
	ros-noetic-joint* \
	ros-noetic-graph* \
	ros-noetic-ros-controller* \
	ros-noetic-joy-teleop* \
	ros-noetic-eigen* \
	ros-noetic-rosbridge-server* \
	ros-noetic-geometric* \
	ros-noetic-object-recognition* \
	ros-noetic-map-server* \
	ros-noetic-warehouse-ros* \
	ros-noetic-geodesy && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "export ROSLAUNCH_SSH_UNKNOWN=1" >> /root/.bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "export ROS_HOSTNAME=localhost" >> /root/.bashrc
RUN echo "export ROS_IP=localhost" >> /root/.bashrc
RUN echo "alias cm='cd /home/$USER/catkin_ws && catkin_make'" >> /root/.bashrc
RUN echo "alias sourceros='source ~/catkin_ws/devel/setup.bash'" >> /root/.bashrc


#####################################################
# OpenAI gym
#####################################################
RUN sudo git clone https://github.com/openai/gym.git && \
	cd gym && \
	git checkout tags/0.22.0 && \
	sudo python3 setup.py install 
COPY pip/requirements_gym.txt requirements_gym.txt
RUN pip install -r requirements_gym.txt


#####################################################
# RL
#####################################################
COPY pip/requirements_rl.txt requirements_rl.txt
RUN pip install -r requirements_rl.txt


#####################################################
# BMIL
#####################################################
COPY pip/requirements_bmil.txt requirements_bmil.txt
RUN pip install -r requirements_bmil.txt
RUN git clone https://github.com/openai/baselines.git && \
	cd baselines && \
	pip install -e .


#####################################################
# Pytorch
#####################################################
# RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

#####################################################
# Run scripts (commands)
#####################################################

WORKDIR /home/

RUN echo 'export DISPLAY=:1' >> ~/.bashrc