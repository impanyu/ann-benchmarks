FROM ann-benchmarks

RUN apt-get update
RUN apt-get install -y wget  python3-setuptools
RUN pip3 install pybind11 numpy


RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format libboost-dev libboost-program-options-dev libmkl-full-dev libcpprest-dev python3.10 pip virtualenv

RUN  rm -rf /var/lib/apt/lists/*

RUN pip3 install build


RUN git clone https://github.com/impanyu/high_order_DiskANN.git
RUN python3 -m high_order_DiskANN/build


RUN pip3 install dist/diskannpy-0.7.0rc2-cp310-cp310-linux_x86_64.whl

RUN python3 -c 'import diskannpy'