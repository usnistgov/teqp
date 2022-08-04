## Just use docker-compose up to run

FROM ubuntu:22.04

RUN apt-get -y -m update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake g++ git zip nano gcovr ninja-build

# This ADD block forces a build (invalidates the cache) if the git repo contents have changed, otherwise leaves it untouched.
# See https://stackoverflow.com/a/39278224
ADD https://api.github.com/repos/usnistgov/teqp/git/refs/heads/master teqp-version.json
RUN git clone --recursive --shallow-submodules https://github.com/usnistgov/teqp

COPY CodeCoverage.cmake /teqp

# Run the catch exe, generating gcov output
WORKDIR /teqp
CMD mkdir build && cd build && cp ../CodeCoverage.cmake . && \
    cmake .. -GNinja -DTEQP_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug -DTEQP_NO_PYTHON=ON && \
    cmake --build . --target teqp_coverage && \
    zip -r /output/html.zip teqp_coverage