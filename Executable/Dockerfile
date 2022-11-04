FROM mambaorg/micromamba:1.0.0
# FROM python:3.9.7
LABEL software.version="0.0.0"
LABEL image.name="linfengwang/tb-ml-drug-resistance-prediction-from-one-hot-encoded-consensus-sequence"
ADD main.py .
COPY --chown=$MAMBA_USER:$MAMBA_USER spec-file.txt /tmp/spec-file.txt
RUN micromamba install -y -n base --file /tmp/spec-file.txt && \
    micromamba clean --all --yes
# # create a directory for the internal data used by the container
# USER root
# RUN mkdir /internal_data /data
# copy the python main and bash entrypoint scripts
# set `/data` as working directory so that the output is written to the
# mount point when run with `docker run -v $PWD:/data ... -o output.csv`
# WORKDIR /data
CMD ["python", "main.py"]