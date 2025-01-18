FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd --gid $GROUP_ID user && \
    useradd --uid $USER_ID --gid $GROUP_ID --create-home user

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

USER user:user
WORKDIR /app

COPY requirements.txt .

USER root

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir optimum
RUN pip install --no-cache-dir auto-gptq

USER user:user
COPY lib lib
COPY models models
COPY cli.py cli.py

ENTRYPOINT ["python", "cli.py"]
