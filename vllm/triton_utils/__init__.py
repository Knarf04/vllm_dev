# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils.importing import (HAS_TRITON, TritonLanguagePlaceholder,
                                         TritonPlaceholder)

__all__ = ["HAS_TRITON", "triton", "tl"]

if HAS_TRITON:
    import triton
    import triton.language as tl
    from vllm.triton_utils.custom_cache_manager import (
        maybe_set_triton_cache_manager)
    
    __all__ += ["maybe_set_triton_cache_manager"]

else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()

