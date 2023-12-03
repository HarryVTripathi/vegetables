FROM python:3.10-slim-bullseye

WORKDIR /
COPY app ./app/
COPY requirements.txt ./requirements.txt

# RUN apt-get update && apt-get -y install gnupg
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9 6ED0E7B82643E131 F8D2585B8783D481
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 54404762BBB6E853 BDE6D2B9216EC7A8

RUN apt-get -y  update && apt-get -y install --no-install-recommends git-all git-lfs 
RUN apt-get -y autoremove && apt-get clean

# RUN apk update && apk add git  # alpine
RUN git clone https://huggingface.co/herrsch99/vegetables

RUN pip install -r requirements.txt

EXPOSE 8084

CMD ["python", "./app/app.py"]