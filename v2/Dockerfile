FROM "ubuntu"


RUN apt-get update && yes | apt-get upgrade

RUN mkdir -p /src

RUN apt-get install -y git python-pip python-dev build-essential
RUN pip install --upgrade pip

RUN pip install tensorflow

RUN apt-get install -y protobuf-compiler python-pil python-lxml
RUN pip install matplotlib

ADD model /src
ADD server /src
WORKDIR /src
