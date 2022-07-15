ARG WORKDIR=/opt/mama.ai

FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-scipy:v1.5.0 as build
ARG WORKDIR

WORKDIR ${WORKDIR}
USER root

# prepare a base layer
RUN true \
    # remove clutter from the python image
    && rm -rf /tmp/* \
    # a few basic debugging utils
    && apt-get update \
    && apt-get install -y --no-install-recommends libxrender1 libxext6 \
    && apt-get install -y --no-install-recommends busybox curl procps \
    && rm -rf /var/lib/apt/lists/* \
    && busybox --install -s \
    && true \


# create WORKDIR
RUN true \
    && mkdir -p "$WORKDIR" \
    && chown -R ${NB_USER} "$WORKDIR" \
    && true

# Conda dependencies
#COPY requirements/environment.yml requirements/
#RUN conda env update -f requirements/environment.yml

COPY requirements/pip_requirements.txt requirements/pip_requirements.txt
RUN python3 -m pip install -r requirements/pip_requirements.txt --quiet --no-cache-dir
RUN python3 -m pip install git+https://github.com/bp-kelley/descriptastorus git+https://github.com/the-mama-ai/selecting_OOD_detector



