# SPDX-License-Identifier: Apache-2.0
"""Preserve Prism audio outputs across prefill graph replay."""

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)


class PrismPrefillCudaGraphRunner(PrefillCudaGraphRunner):
    def _trim_logits_output(  # noqa: leading-underscore - SGLang runner override
        self, output: LogitsProcessorOutput
    ) -> LogitsProcessorOutput:
        result = super()._trim_logits_output(output)
        result.customized_info = output.customized_info
        return result
