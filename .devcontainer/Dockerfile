FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install pyenv
# v2.3.6 is the latest release as of 2022-11-14
RUN git clone --depth 1 --single-branch --branch v2.3.6 https://github.com/pyenv/pyenv.git /root/.pyenv \
    && echo 'eval "$(pyenv init --path)"' >> /root/.zshrc
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

# Install mlflower dependencies
RUN pip install --no-cache-dir mlflow virtualenv kubernetes pipx && pipx install tldr

# MLFlow UI
WORKDIR /mlflow
# ENV MLFLOW_SERVER_URI sqlite:///${MLFLOW_HOME}/mlflow.db
ENV MLFLOW_TRACKING_URI="http://localhost:5000"

# Start MLflow UI
EXPOSE 5000
CMD mlflow ui --host 0.0.0.0 --port 5000
