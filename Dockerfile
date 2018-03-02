FROM python:3
COPY . /app
WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends -y \
    mecab \
    libmecab-dev \
    mecab-ipadic \
    mecab-ipadic-utf8 \
    && apt-get autoclean; apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install mecab-python3

RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && \
    mkdir /usr/lib/mecab/dic && \
    /app/mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y

ENTRYPOINT bash
