## Just use docker-compose up to run

FROM ubuntu:24.04

RUN apt-get -y -m update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake g++ git zip nano valgrind ninja-build python3

# This ADD block forces a build (invalidates the cache) if the git repo contents have changed, otherwise leaves it untouched.
# See https://stackoverflow.com/a/39278224
ADD https://api.github.com/repos/usnistgov/teqp/git/refs/heads/master teqp-version.json
RUN git clone --recursive --shallow-submodules https://github.com/usnistgov/teqp

# Build the catch exe
WORKDIR /teqp
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTEQP_NO_PYTHON=ON && \
    cmake --build . --target catch_tests

COPY run_valgrind.py /
WORKDIR /
CMD python3 -u run_valgrind.py
