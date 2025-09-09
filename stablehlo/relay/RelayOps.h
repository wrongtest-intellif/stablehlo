/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_RELAY_OPS_H
#define STABLEHLO_RELAY_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {
namespace relay {

// TVM Relay operation implementations
// Start with a single binary operation as requested

/// TVM Relay add operation - element-wise addition of two tensors
/// This mirrors the functionality of StableHLO addOp but provides
/// a TVM Relay interface and implementation
Tensor addOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);

}  // namespace relay
}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_RELAY_OPS_H