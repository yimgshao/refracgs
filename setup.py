# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- Modification Notice ---
# Modified by Yiming Shao
# Description:
# Renamed the package metadata for the RefracGS project.

from setuptools import find_packages, setup

VERSION = '0.0.1'

setup(
    name="refracgs",
    version=VERSION,
    author="Yiming Shao, et al.",
    author_email="yimingshao1206@gmail.com",
    description="Official implementation of the ECCV'26 paper RefracGS.",
    long_description_content_type="text/markdown",
    py_modules=["threedgrut", "threedgrt_tracer", "threedgut_tracer", "playground", "water_tracer"],
    packages=find_packages(where='libs'),
    python_requires=">=3.11",
    install_requires=[],
    classifiers=["Operating System :: OS Independent"],
)
