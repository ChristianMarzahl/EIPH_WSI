FROM ubuntu:20.04 as builder

WORKDIR /usr/src/EIPH_WSI
ENV DEBIAN_FRONTEND="noninteractive" TZ="SystemV"


RUN apt-get update && apt-get install -y python3-pip python3-openslide python3-opencv  libvips libvips-dev

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip3 install --upgrade pip
COPY ./requirements.txt /usr/src/EIPH_WSI/requirements.txt
RUN pip3 install -r requirements.txt

COPY . /usr/src/EIPH_WSI/


# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]


CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

