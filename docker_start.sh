#!/bin/bash
# to run the file command: bash docker_start.sh

docker run --rm -it --runtime=nvidia --privileged --net=host \
--ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY \
-v $HOME/.Xauthority:/home/$(id -un)/.Xauthority \
-e XAUTHORITY=/home/$(id -un)/.Xauthority \
-e DOCKER_USER_NAME=$(id -un) \
-e DOCKER_USER_ID=$(id -u) \
-e DOCKER_USER_GROUP_NAME=$(id -gn) \
-e DOCKER_USER_GROUP_ID=$(id -g) \
-v $(pwd)/next_word_predictor:/home/vaishanth/next_word_predictor \
ml