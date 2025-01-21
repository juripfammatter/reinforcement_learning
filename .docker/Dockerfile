FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /home/

# setup pip venv
ARG VENV=/home/.venv
RUN python3 -m venv /${VENV}

ENV VIRTUAL_ENV=${VENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
#RUN /bin/bash -c "python3 -m venv /${VENV} \
#    && source ${VENV}/bin/activate \
#    && python3 -m pip install --upgrade pip"
RUN python3 -m pip install --upgrade pip

# install requirements
ARG REQUIREMENTS_FILE
COPY ${REQUIREMENTS_FILE} /home/requirements.txt

#RUN /bin/bash -c "source ${VENV}/bin/activate\
#    && pip install -r /home/requirements.txt "
RUN pip install -r /home/requirements.txt