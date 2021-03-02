#!/bin/bash

build_dir="/tmp/tritonbuild"
build_branch=mlperf-inference-v1.0
tf2_base="nvcr.io/nvidia/tensorflow:21.02-tf2-py3"
tf2_docker_image=tritonserver_tf2_cpu

mkdir -p ${build_dir}
cd ${build_dir}
rm -fr *

# Write Dockerfile needed to build the patched version of TF2
cat >Dockerfile.${tf2_docker_image} <<EOF
FROM ${tf2_base}
WORKDIR /opt/tensorflow

# update build options to enable some CPU options
RUN echo "-c opt --config=mkl --config=cuda --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 --output_filter MATCH_NOTHING --define build_with_mkl_dnn_v1_only=true --copt=-DENABLE_INTEL_MKL_BFLOAT16 --copt=-O3" >nvbuildopts
RUN cat nvbuildopts

RUN ./nvbuild.sh --triton --v2
EOF

# Docker build of patched TF2.
docker build --pull -t ${tf2_docker_image} -f Dockerfile.${tf2_docker_image} .

# Clone
cd ${build_dir}
git clone --single-branch --depth=1 -b ${build_branch} https://github.com/triton-inference-server/server.git

# Build for CPU, TensorFlow2 and OpenVINO backends only
cd ${build_dir}
rm -fr build
mkdir build

cd server
./build.py -v --image=tensorflow2,${tf2_docker_image} --build-dir=${build_dir}/build --cmake-dir=/workspace/build --enable-logging --enable-stats --repo-tag=common:${build_branch} --repo-tag=core:${build_branch} --repo-tag=backend:${build_branch} --repo-tag=thirdparty:${build_branch} --endpoint=grpc --endpoint=http --backend=tensorflow2:${build_branch} --backend=openvino:${build_branch}
