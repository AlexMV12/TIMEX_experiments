# syntax=docker/dockerfile:1
FROM python:3.8-slim
RUN apt-get update \
&& apt-get install gcc g++ build-essential python-dev -y \
&& apt-get clean

WORKDIR /app
COPY . .
RUN /app/install_timex_venv.sh
RUN /app/install_darts_venv.sh

