FROM python:3.10-buster

COPY  requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# set working dir for the project folder
WORKDIR /opt/project