FROM python:3.9-buster

# Install dependencies
RUN pip install openai \
    latexify-py sympy  \
    pandas numpy matplotlib \
    jupyter nbconvert ipywidgets \
    gpt-engineer

# set working dir for the project folder
WORKDIR /opt/project