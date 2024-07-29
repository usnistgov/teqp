## Just use docker-compose up to run

# Or, pick a different tag (change the tag) to use a different version of clang
FROM silkeh/clang:18

RUN apt-get -y -m update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake git zip nano gcovr ninja-build

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 libpython3-dev python3-distutils

# Run the catch exe, generating gcov output
CMD mkdir build && cd build && \
    cmake /teqp -GNinja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --target teqpcpp --parallel 4
