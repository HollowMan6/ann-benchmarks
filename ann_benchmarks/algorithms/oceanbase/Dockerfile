FROM ann-benchmarks

RUN git clone https://github.com/HollowMan6/oceanbase /tmp/oceanbase
WORKDIR /tmp/oceanbase
RUN git checkout vector_search

RUN apt-get update && apt-get -y install bash wget rpm rpm2cpio cpio make build-essential binutils m4
RUN pip install PyMySQL
RUN bash build.sh release --init --make
RUN bash tools/deploy/obd.sh prepare -p /tmp/obtest
RUN bash tools/deploy/obd.sh deploy -c ./tools/deploy/single.yaml
