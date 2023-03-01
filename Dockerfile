FROM python:3.8.5

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN apt-get update

RUN python -m pip --no-cache-dir install --upgrade pip

RUN pip install poetry && \
    poetry config virtualenvs.create false

WORKDIR /project
COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root --only main
