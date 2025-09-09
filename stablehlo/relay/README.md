# TVM Relay Operations for StableHLO

This subdirectory contains TVM Relay style operation implementations that mirror HLO operations but provide TVM Relay semantics and interfaces.

## Overview

The relay module provides:

1. **C++ Implementation** (`RelayOps.h`, `RelayOps.cpp`): Core TVM Relay operation implementations
2. **Python Bindings** (`python/`): Python interface for TVM Relay operations  
3. **Build Integration**: CMake and Bazel build configurations

## Architecture

### C++ Layer
- `RelayOps.h`: Header file defining TVM Relay operation interfaces
- `RelayOps.cpp`: Implementation of TVM Relay operations using StableHLO infrastructure

### Python Layer  
- `python/relay_ops.py`: Pure Python implementation of TVM Relay operations
- `python/__init__.py`: Module initialization and exports

## Current Operations

### addOp
Element-wise addition of two tensors following TVM Relay semantics.

**C++ Interface:**
```cpp
Tensor addOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
```

**Python Interface:**
```python
def add(lhs: RelayTensor, rhs: RelayTensor) -> RelayTensor
```

## Usage

### Python Example
```python
from stablehlo.relay.python import RelayTensor, add

# Create tensors
tensor1 = RelayTensor([1, 2, 3])
tensor2 = RelayTensor([4, 5, 6])

# Perform addition
result = add(tensor1, tensor2)
print(result.data)  # [5, 7, 9]
```

## Testing

Run the Python tests:
```bash
cd stablehlo/relay
python test_relay_ops.py
```

## Future Extensions

This is designed to be easily extensible. Additional TVM Relay operations can be added by:

1. Adding C++ function declarations to `RelayOps.h`
2. Implementing the functions in `RelayOps.cpp`
3. Adding corresponding Python functions to `python/relay_ops.py`
4. Updating the exports in `python/__init__.py`

## Design Principles

- **Minimal Changes**: Reuses existing StableHLO infrastructure where possible
- **TVM Relay Semantics**: Operations follow TVM Relay conventions and behaviors
- **Extensible**: Easy to add new operations following established patterns
- **Testable**: Simple testing infrastructure for verification