DATA_FOLDER="/media/TrainDataset/orbslam"

# checking if you have nvidia
if ! nvidia-smi | grep "Driver" 2>/dev/null; then
  echo "******************************"
  echo """It looks like you don't have nvidia drivers running. Consider running build_container_cpu.sh instead."""
  echo "******************************"
  while true; do
    read -p "Do you still wish to continue?" yn
    case $yn in
      [Yy]* ) make install; break;;
      [Nn]* ) exit;;
      * ) echo "Please answer yes or no.";;
    esac
  done
fi 

# UI permisions
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

xhost +local:docker
#DOCKER_NAME=jahaniam/orbslam3:ubuntu20_noetic_cuda
# DOCKER_NAME=dshong/orbslam:2-ubuntu20_noetic_cuda
DOCKER_NAME=dshong/dynaslam:ubuntu20_noetic_cuda
NAME=dyanslam

# Remove existing container
docker rm -f $NAME &>/dev/null

# Create a new container
docker run -ti --privileged --net=host --ipc=host \
    --name=$NAME \
    --gpus=all \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e "XAUTHORITY=$XAUTH" \
    -e ROS_IP=127.0.0.1 \
    --cap-add=SYS_PTRACE \
    -v `pwd`:/ws/external \
    -v $DATA_FOLDER:/ws/data \
    -v /etc/group:/etc/group:ro \
    $DOCKER_NAME bash

