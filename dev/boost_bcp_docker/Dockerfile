## Use docker-compose to spin up this job

FROM ubuntu:20.04

RUN apt-get -y -m update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake g++ git zip wget

RUN mkdir /boost && \
	wget -c https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz -O - | tar -xz -C /boost && \
	cd /boost/boost_1_77_0/ && \
	./bootstrap.sh && \
	./b2 tools/bcp && \
	mkdir /boost_teqp && \
	bin.v2/tools/bcp/gcc-9/release/link-static/bcp multiprecision/cpp_bin_float.hpp multiprecision/eigen.hpp functional/hash.hpp numeric/odeint.hpp /boost_teqp && \
	zip -r /boost_teqp.zip /boost_teqp && \
	tar czf /boost_teqp.tar.gz /boost_teqp

CMD cp /*.tar.gz /output