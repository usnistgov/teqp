## Use docker-compose to spin up this job

FROM ubuntu:24.04

RUN apt-get -y -m update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake g++ git zip wget xz-utils

RUN mkdir /boost && \

	wget -c --no-check-certificate https://archives.boost.io/release/1.88.0/source/boost_1_88_0.tar.gz  -O - | tar -xz -C /boost && \
	cd /boost/boost_1_88_0/ && \
	./bootstrap.sh && \
	./b2 tools/bcp

WORKDIR /boost/boost_1_88_0
RUN mkdir /boost_teqp && \
	dist/bin/bcp multiprecision/cpp_bin_float.hpp multiprecision/cpp_complex.hpp multiprecision/eigen.hpp functional/hash.hpp numeric/odeint.hpp typeof/incr_registration_group.hpp mp11.hpp algorithm/string/join.hpp asio/thread_pool.hpp asio/post.hpp math/tools/toms748_solve.hpp /boost_teqp && \
	zip -r /boost_teqp.zip /boost_teqp &&  \
	tar cJf /boost_teqp.tar.xz /boost_teqp	

CMD cp /*.tar.xz /output
