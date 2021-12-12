FROM nvidia/cuda:10.2-base

USER root

RUN apt-get update

RUN apt-get install -y python3.7 python3-pip

RUN apt-get -y install git cmake

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir matplotlib pillow numpy tqdm

RUN pip3 install torch torchvision torchaudio

RUN pip3 install av aiortc joblib scipy compressai pytorch_msssim

COPY ./ ./

CMD ["python3", "streaming.py"]