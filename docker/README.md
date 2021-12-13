# build the docker images
sudo docker build -t client -f docker/Dockerfile.client .
sudo docker build -t server -f docker/Dockerfile.server .

# start the docker container
1. On server
sudo docker run -it --rm --gpus all -p 8888:8888/tcp server
2. On client
sudo docker run -it --rm -p 8888:8888/tcp client

sudo docker build -t benchmark -f docker/Dockerfile .

sudo docker run -it --rm --gpus all -p 8888:8888 benchmark

sudo docker run -it --rm --gpus all --network host benchmark