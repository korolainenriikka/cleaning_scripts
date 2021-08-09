FROM continuumio/miniconda3:latest

RUN pip install mlflow==1.17.0 \
  python-dateutil==2.8.1 \
  geopandas==0.9.0 \
  pandas==1.3.0 \
  numpy==1.20.1 \
  numpy-ext==0.9.4 \
  numpydoc==0.7.0 \
  matplotlib==3.4.2

