"""
Dual License

For Open-Source Individuals:
MIT License

Copyright (c) 2025 James B. Chapman

Permission is hereby granted, free of charge, to any individual obtaining a copy
of this software and associated documentation files (the "Software"), for personal,
non-commercial use, to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For Companies:
Commercial use by companies requires a separate license. Contact iconoclastdao@gmail.com for licensing terms and conditions. Unauthorized commercial use is prohibited.
"""

import json
import time
import torch
import numpy as np
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
from .tcc_logger import TCCLogger
from ..modules.security import BioInspiredSecuritySystem

class LLMModule(ABC):
    def __init__(self, logger: TCCLogger = None):
        self.logger = logger or TCCLogger()

    @abstractmethod
    def compute(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    def reverse(self, output_data: Any) -> Any:
        pass

    @abstractmethod
    def mimic_transformation(self, input_data: Any, ref_input: Any, ref_output: Any) -> Any:
        pass

class ModelManager:
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

class TokenizerModule(LLMModule):
    def __init__(self, model_manager: ModelManager, logger: TCCLogger = None, security: BioInspiredSecuritySystem = None):
        super().__init__(logger)
        self.model_manager = model_manager
        self.tokenizer = model_manager.tokenizer
        self.security = security

    def compute(self, input_text: str) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            if self.security:
                is_anomaly, score, explanation = self.security.detect_anomaly(input_text.encode())
                if is_anomaly:
                    self.logger.log(
                        "tokenize_anomaly",
                        input_text.encode(),
                        b"",
                        {"score": score, "explanation": explanation, "execution_time_ns": time.time_ns() - start_time},
                        "ERROR",
                        "ANOMALY_DETECTED"
                    )
                    raise ValueError(f"Anomaly detected: {explanation}")
            tokens = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            token_ids = tokens["input_ids"].numpy().tobytes()
            attention_mask = tokens["attention_mask"].numpy().tobytes()
            output = {"input_ids": token_ids, "attention_mask": attention_mask}
            self.logger.log(
                "tokenize",
                input_text.encode(),
                json.dumps(output).encode(),
                {"token_count": len(tokens["input_ids"][0]), "execution_time_ns": time.time_ns() - start_time}
            )
            return output
        except Exception as e:
            self.logger.log(
                "tokenize",
                input_text.encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "TOKENIZE_FAILED"
            )
            raise

    def reverse(self, output_data: Dict[str, Any]) -> str:
        start_time = time.time_ns()
        try:
            input_ids = np.frombuffer(output_data["input_ids"], dtype=np.int64).reshape(1, -1)
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            self.logger.log(
                "reverse_tokenize",
                json.dumps(output_data).encode(),
                text.encode(),
                {"execution_time_ns": time.time_ns() - start_time}
            )
            return text
        except Exception as e:
            self.logger.log(
                "reverse_tokenize",
                json.dumps(output_data).encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "REVERSE_TOKENIZE_FAILED"
            )
            raise

    def mimic_transformation(self, input_text: str, ref_input: str, ref_output: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            adjusted_text = input_text[:len(ref_input)] if len(input_text) > len(ref_input) else input_text + " " * (len(ref_input) - len(input_text))
            output = self.compute(adjusted_text)
            self.logger.log(
                "mimic_tokenize",
                input_text.encode(),
                json.dumps(output).encode(),
                {"ref_input": ref_input, "adjusted_text": adjusted_text, "execution_time_ns": time.time_ns() - start_time}
            )
            return output
        except Exception as e:
            self.logger.log(
                "mimic_tokenize",
                input_text.encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "MIMIC_TOKENIZE_FAILED"
            )
            raise

class EmbedderModule(LLMModule):
    def __init__(self, model_manager: ModelManager, logger: TCCLogger = None):
        super().__init__(logger)
        self.model = model_manager.model
        self.embedding_layer = self.model.get_input_embeddings()

    def compute(self, tokens: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            input_ids = torch.from_numpy(np.frombuffer(tokens["input_ids"], dtype=np.int64).reshape(1, -1))
            embeddings = self.embedding_layer(input_ids).detach().numpy().tobytes()
            output = {"embeddings": embeddings, "input_ids": tokens["input_ids"]}
            self.logger.log(
                "embed",
                tokens["input_ids"],
                embeddings,
                {"shape": str(np.frombuffer(embeddings).shape), "execution_time_ns": time.time_ns() - start_time}
            )
            return output
        except Exception as e:
            self.logger.log(
                "embed",
                tokens["input_ids"],
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "EMBED_FAILED"
            )
            raise

    def reverse(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            input_ids = output_data["input_ids"]
            self.logger.log(
                "reverse_embed",
                output_data["embeddings"],
                input_ids,
                {"execution_time_ns": time.time_ns() - start_time}
            )
            return {"input_ids": input_ids}
        except Exception as e:
            self.logger.log(
                "reverse_embed",
                output_data["embeddings"],
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "REVERSE_EMBED_FAILED"
            )
            raise

    def mimic_transformation(self, tokens: Dict[str, Any], ref_input: Dict[str, Any], ref_output: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            output = self.compute(tokens)
            self.logger.log(
                "mimic_embed",
                tokens["input_ids"],
                output["embeddings"],
                {"ref_input_ids": ref_input["input_ids"].hex(), "execution_time_ns": time.time_ns() - start_time}
            )
            return output
        except Exception as e:
            self.logger.log(
                "mimic_embed",
                tokens["input_ids"],
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "MIMIC_EMBED_FAILED"
            )
            raise

class TransformerLayerModule(LLMModule):
    def __init__(self, model_manager: ModelManager, layer_idx: int = 0, logger: TCCLogger = None):
        super().__init__(logger)
        self.model = model_manager.model
        self.layer = self.model.transformer.h[layer_idx]
        self.layer_idx = layer_idx

    def compute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            embeddings = torch.from_numpy(np.frombuffer(input_data["embeddings"]).reshape(1, -1, self.model.config.hidden_size))
            input_ids = input_data["input_ids"]
            attention_mask = torch.from_numpy(np.frombuffer(input_data.get("attention_mask", b""), dtype=np.int64).reshape(1, -1))
            outputs = self.layer(embeddings, attention_mask=attention_mask)[0].detach().numpy().tobytes()
            output = {"hidden_states": outputs, "input_ids": input_ids}
            self.logger.log(
                "transformer_layer",
                input_data["embeddings"],
                outputs,
                {"layer_idx": self.layer_idx, "execution_time_ns": time.time_ns() - start_time}
            )
            return output
        except Exception as e:
            self.logger.log(
                "transformer_layer",
                input_data["embeddings"],
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "TRANSFORMER_LAYER_FAILED"
            )
            raise

    def reverse(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            input_ids = output_data["input_ids"]
            self.logger.log(
                "reverse_transformer_layer",
                output_data["hidden_states"],
                input_ids,
                {"layer_idx": self.layer_idx, "execution_time_ns": time.time_ns() - start_time}
            )
            return {"embeddings": output_data["hidden_states"], "input_ids": input_ids}
        except Exception as e:
            self.logger.log(
                "reverse_transformer_layer",
                output_data["hidden_states"],
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "REVERSE_TRANSFORMER_LAYER_FAILED"
            )
            raise

    def mimic_transformation(self, input_data: Dict[str, Any], ref_input: Dict[str, Any], ref_output: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            output = self.compute(input_data)
            self.logger.log(
                "mimic_transformer_layer",
                input_data["embeddings"],
                output["hidden_states"],
                {"layer_idx": self.layer_idx, "ref_input_ids": ref_input["input_ids"].hex(), "execution_time_ns": time.time_ns() - start_time}
            )
            return output
        except Exception as e:
            self.logger.log(
                "mimic_transformer_layer",
                input_data["embeddings"],
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "MIMIC_TRANSFORMER_LAYER_FAILED"
            )
            raise

class DecoderModule(LLMModule):
    def __init__(self, model_manager: ModelManager, temperature: float = 1.0, top_k: int = 50, logger: TCCLogger = None):
        super().__init__(logger)
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer
        self.temperature = temperature
        self.top_k = top_k

    def compute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            hidden_states = torch.from_numpy(np.frombuffer(input_data["hidden_states"]).reshape(1, -1, self.model.config.hidden_size))
            input_ids = torch.from_numpy(np.frombuffer(input_data["input_ids"], dtype=np.int64).reshape(1, -1))
            logits = self.model.lm_head(hidden_states).detach()
            probs = torch.softmax(logits / self.temperature, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
            token_idx = torch.multinomial(top_k_probs[0, -1], 1).item()
            next_token = top_k_indices[0, -1, token_idx].item()
            output_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
            output = {"output_text": output_text, "logits": logits.numpy().tobytes(), "next_token": next_token}
            self.logger.log(
                "decode",
                input_data["hidden_states"],
                output_text.encode(),
                {"next_token": next_token, "execution_time_ns": time.time_ns() - start_time}
            )
            return output
        except Exception as e:
            self.logger.log(
                "decode",
                input_data["hidden_states"],
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "DECODE_FAILED"
            )
            raise

    def reverse(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            logits = np.frombuffer(output_data["logits"]).reshape(1, -1, self.model.config.vocab_size)
            hidden_states = self.model.lm_head.weight.data.T @ torch.from_numpy(logits).squeeze(0).T
            self.logger.log(
                "reverse_decode",
                output_data["output_text"].encode(),
                hidden_states.numpy().tobytes(),
                {"execution_time_ns": time.time_ns() - start_time}
            )
            return {"hidden_states": hidden_states.numpy().tobytes(), "input_ids": output_data.get("input_ids", b"")}
        except Exception as e:
            self.logger.log(
                "reverse_decode",
                output_data["output_text"].encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "REVERSE_DECODE_FAILED"
            )
            raise

    def mimic_transformation(self, input_data: Dict[str, Any], ref_input: Dict[str, Any], ref_output: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        try:
            output = self.compute(input_data)
            self.logger.log(
                "mimic_decode",
                input_data["hidden_states"],
                output["output_text"].encode(),
                {"ref_output_text": ref_output["output_text"], "execution_time_ns": time.time_ns() - start_time}
            )
            return output
        except Exception as e:
            self.logger.log(
                "mimic_decode",
                input_data["hidden_states"],
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "MIMIC_DECODE_FAILED"
            )
            raise

class LLMFlow:
    def __init__(self, steps: List[Tuple[str, LLMModule, Dict[str, Any]]], reference_input: str, logger: TCCLogger = None):
        self.steps = steps
        self.logger = logger or TCCLogger()
        self.flow_log: List[Dict[str, Any]] = []
        self.reference_input = reference_input
        self.reference_outputs = self._compute_reference_outputs()

    def _compute_reference_outputs(self) -> List[Any]:
        outputs = [self.reference_input]
        current_data = self.reference_input
        for step_name, module, _ in self.steps:
            output_data = module.compute(current_data)
            outputs.append(output_data)
            current_data = output_data
        return outputs

    def execute(self, input_text: str) -> str:
        start_time = time.time_ns()
        current_data = input_text
        for step_idx, (step_name, module, params) in enumerate(self.steps):
            try:
                output_data = module.compute(current_data)
                output_bytes = json.dumps(output_data).encode() if isinstance(output_data, dict) else output_data.encode()
                input_bytes = current_data.encode() if isinstance(current_data, str) else json.dumps(current_data).encode()
                metadata = {
                    "step_index": step_idx,
                    "step_name": step_name,
                    "params": params,
                    "output_type": str(type(output_data))
                }
                self.logger.log(
                    f"flow_{step_name}",
                    input_bytes,
                    output_bytes,
                    metadata
                )
                self.flow_log.append({
                    "step_index": step_idx,
                    "step_name": step_name,
                    "operation_id": self.logger.logs[-1].operation_id,
                    "input_data": base64.b64encode(input_bytes).decode(),
                    "output_data": base64.b64encode(output_bytes).decode(),
                    "timestamp": time.time_ns()
                })
                current_data = output_data
            except Exception as e:
                self.logger.log(
                    f"flow_{step_name}",
                    input_bytes,
                    b"",
                    {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                    "ERROR",
                    "FLOW_STEP_FAILED"
                )
                raise
        final_output = current_data["output_text"] if isinstance(current_data, dict) else current_data
        self.logger.log(
            "flow_complete",
            input_text.encode(),
            final_output.encode(),
            {"total_steps": len(self.steps), "execution_time_ns": time.time_ns() - start_time}
        )
        return final_output

    def save_flow_log(self, filename: str) -> None:
        try:
            with open(filename, 'w') as f:
                for entry in self.flow_log:
                    f.write(json.dumps(entry) + '\n')
        except IOError as e:
            self.logger.log(
                "save_flow_log",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "SAVE_LOG_FAILED"
            )
            raise

def define_default_flow(model_name: str = "distilgpt2", num_layers: int = 2, logger: TCCLogger = None, security: BioInspiredSecuritySystem = None) -> LLMFlow:
    reference_input = "Hello, world!"
    model_manager = ModelManager(model_name)
    steps = [
        ("tokenize", TokenizerModule(model_manager, logger, security), {}),
        ("embed", EmbedderModule(model_manager, logger), {}),
    ]
    for i in range(min(num_layers, 6)):
        steps.append((f"transformer_layer_{i}", TransformerLayerModule(model_manager, i, logger), {}))
    steps.append(("decode", DecoderModule(model_manager, logger=logger), {"temperature": 1.0, "top_k": 50}))
    return LLMFlow(steps, reference_input, logger)