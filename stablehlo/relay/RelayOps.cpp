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

#include "stablehlo/relay/RelayOps.h"

#include "stablehlo/reference/Element.h"

namespace mlir {
namespace stablehlo {
namespace relay {

Tensor addOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  // TVM Relay add operation implementation
  // Follows the same pattern as StableHLO addOp but with relay-specific logic
  Tensor result(resultType);
  
  // Element-wise addition across all indices
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    Element lhsElement = lhs.get(*it);
    Element rhsElement = rhs.get(*it);
    result.set(*it, lhsElement + rhsElement);
  }
  
  return result;
}

}  // namespace relay
}  // namespace stablehlo
}  // namespace mlir