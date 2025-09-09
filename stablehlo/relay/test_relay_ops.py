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

"""Simple test for TVM Relay operations."""

import sys
import os
import numpy as np

# Add the relay Python module to the path
relay_python_path = os.path.join(os.path.dirname(__file__), 'python')
sys.path.insert(0, relay_python_path)

try:
    from relay_ops import RelayTensor, add
    
    # Test basic tensor creation
    print("Testing RelayTensor creation...")
    tensor1 = RelayTensor([1, 2, 3])
    tensor2 = RelayTensor([4, 5, 6])
    
    print(f"Tensor 1: {tensor1}")
    print(f"Tensor 2: {tensor2}")
    
    # Test add operation
    print("\nTesting add operation...")
    result = add(tensor1, tensor2)
    print(f"Result: {result}")
    print(f"Result data: {result.data}")
    
    # Verify the result
    expected = np.array([5, 7, 9])
    if np.array_equal(result.data, expected):
        print("✓ Add operation test passed!")
    else:
        print("✗ Add operation test failed!")
        print(f"Expected: {expected}")
        print(f"Got: {result.data}")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the relay Python module is in the correct path")
except Exception as e:
    print(f"Test error: {e}")
    import traceback
    traceback.print_exc()

print("\nTVM Relay operations test completed.")