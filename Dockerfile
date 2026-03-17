FROM python:3.13-slim AS build
RUN apt-get update \
    && apt-get install -y \
    linux-headers-generic\
    gcc \
    g++ \
    gfortran \
    musl-dev \
    libffi-dev \
    git \
    ffmpeg
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

FROM build AS build-venv
COPY requirements.txt ./
RUN pip install --disable-pip-version-check -r requirements.txt && \
    pip cache purge
COPY . .
CMD [ "python", "-u","./main.py" ]