## Just use docker-compose up to run

# Testing on M2 mac, so arch is needed
FROM --platform=linux/amd64 archlinux

RUN pacman -Syu --noconfirm

RUN pacman -S --noconfirm gcc cmake ninja python git

# Run the catch exe, generating gcov output
CMD mkdir build && cd build && \
    cmake /teqp -GNinja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --target teqpcpp --parallel 4
