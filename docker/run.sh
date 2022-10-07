#!/bin/bash

docker run --rm \
  --net=host \
  --ipc=host \
  --gpus all \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:$docker/.Xauthority \
  -v $HOME/unitree-a1-dreamer:$HOME/unitree-a1-dreamer \
  -e XAUTHORITY=$home_folder/.Xauthority \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e DOCKER_USER_NAME=$(id -un) \
  -e DOCKER_USER_ID=$(id -u) \
  -e DOCKER_USER_GROUP_NAME=$(id -gn) \
  -e DOCKER_USER_GROUP_ID=$(id -g) \
  -it --name "unitree-a1-gym" unitree-a1-gym
