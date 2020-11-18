FROM ubuntu:16.04
WORKDIR /app
RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y dbus python3-gi python3-pip psmisc dbus-x11
RUN python3 --version
RUN pip3 --version
ADD regression_ml.cpython-35m-x86_64-linux-gnu.so /app
ADD classification_ml.cpython-35m-x86_64-linux-gnu.so /app
ADD enrouting.py /app
ADD requirements.txt /app
ADD app.py /app
RUN mkdir datasets
RUN mkdir models
RUN mkdir test_input
RUN mkdir test_output
RUN mkdir training_predictions
RUN pip3 install -r requirements.txt
EXPOSE 8000
CMD [ "python3", "./app.py" ]
