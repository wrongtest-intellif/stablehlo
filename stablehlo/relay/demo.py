#!/usr/bin/env python3
# Copyright 2023 The StableHLO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
StableHLO TVM Relay Operations Demo

This script demonstrates the TVM Relay operations implementation
that provides a mirrored version of HLO ops for TVM Relay.
"""

import sys
import os

# Add the relay Python module to the path
relay_python_path = os.path.join(os.path.dirname(__file__), 'python')
sys.path.insert(0, relay_python_path)

from relay_ops import RelayTensor, add

def main():
    print("=" * 60)
    print("StableHLO TVM Relay Operations Demo")
    print("=" * 60)
    
    print("\n1. Creating TVM Relay Tensors:")
    print("-" * 30)
    
    # Test with different data types and shapes
    tensor_int = RelayTensor([1, 2, 3, 4])
    tensor_float = RelayTensor([1.5, 2.5, 3.5, 4.5])
    tensor_2d = RelayTensor([[1, 2], [3, 4]])
    
    print(f"Integer tensor: {tensor_int}")
    print(f"Float tensor: {tensor_float}")
    print(f"2D tensor: {tensor_2d}")
    
    print("\n2. TVM Relay Add Operation:")
    print("-" * 30)
    
    # Demonstrate add operation
    a = RelayTensor([10, 20, 30])
    b = RelayTensor([1, 2, 3])
    result = add(a, b)
    
    print(f"Tensor A: {a.data}")
    print(f"Tensor B: {b.data}")
    print(f"A + B = {result.data}")
    
    print("\n3. Floating Point Operations:")
    print("-" * 30)
    
    x = RelayTensor([1.1, 2.2, 3.3])
    y = RelayTensor([0.9, 1.8, 2.7])
    float_result = add(x, y)
    
    print(f"Tensor X: {x.data}")
    print(f"Tensor Y: {y.data}")
    print(f"X + Y = {float_result.data}")
    
    print("\n4. Architecture Overview:")
    print("-" * 30)
    print("✓ C++ Implementation: RelayOps.h, RelayOps.cpp")
    print("✓ Python Bindings: relay_ops.py")
    print("✓ Build Integration: CMakeLists.txt, BUILD.bazel")
    print("✓ Testing: test_relay_ops.py")
    print("✓ Documentation: README.md")
    
    print("\n" + "=" * 60)
    print("TVM Relay operations successfully implemented!")
    print("Ready for extension with additional operations.")
    print("=" * 60)

if __name__ == "__main__":
    main()