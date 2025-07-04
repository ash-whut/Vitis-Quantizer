#!/bin/bash

# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Daniele Bagni, Xilinx Inc


# clean folders
echo " "
echo "-----------------------------------------"
echo "CLEANING FOLDERS."
echo "-----------------------------------------"

rm -r ./build/log*
mkdir ./build/log

rm    ./build/data/*.pt
rm    ./build/data/*.tar.gz
rm -r ./build/compiled_model
rm -r ./build/quantized_model

rm -r ./build/target*
mkdir ./build/target
mkdir ./build/target/test_images
