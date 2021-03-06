FROM tensorflow/tensorflow:latest-py3

RUN apt update && apt install -y curl \
    && echo "Pulling watchdog binary from Github." \
    && curl -sSLf https://github.com/openfaas-incubator/of-watchdog/releases/download/0.2.5/of-watchdog > /usr/bin/fwatchdog \
    && chmod +x /usr/bin/fwatchdog

RUN pip3 install --upgrade pip \
    && pip3 install flask \
    && pip3 install pyyaml


COPY models models
COPY styles styles
COPY AvatarNet AvatarNet
COPY AvatarNet_config.yml .
COPY handler.py .
COPY run_server.py .

ENV INTER_WEIGHT=0.5
ENV STYLE_PATH="styles/gold.jpg"
ENV write_debug="true"
ENV fprocess="python3 run_server.py"
ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"
ENV exec_timeout=600

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1
CMD [ "fwatchdog" ]
