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
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

For Companies:
Commercial use by companies requires a separate license. Contact iconoclastdao@gmail.com
for licensing terms and conditions. Unauthorized commercial use is prohibited.
"""



from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import json
import time
import hashlib
import base64
import struct
import nacl.signing
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TCCLogEntry:
    step: int
    operation: str
    input_data: bytes
    output_data: bytes
    metadata: Dict[str, Any]
    log_level: str
    error_code: str
    prev_hash: bytes
    operation_id: str
    timestamp: int
    execution_time_ns: int
    signature: bytes

    def _to_bytes_without_signature(self) -> bytes:
        step_bytes = struct.pack('>Q', self.step)
        op_bytes = self.operation.encode('utf-8').ljust(32, b'\x00')[:32]
        input_len_bytes = struct.pack('>I', len(self.input_data))
        output_len_bytes = struct.pack('>I', len(self.output_data))
        meta_bytes = json.dumps(self.metadata).encode('utf-8').ljust(128, b'\x00')[:128]
        level_bytes = self.log_level.encode('utf-8').ljust(16, b'\x00')[:16]
        error_bytes = self.error_code.encode('utf-8').ljust(16, b'\x00')[:16]
        op_id_bytes = self.operation_id.encode('utf-8').ljust(32, b'\x00')[:32]
        ts_bytes = struct.pack('>q', self.timestamp)
        exec_time_bytes = struct.pack('>q', self.execution_time_ns)
        return (
            step_bytes + op_bytes + input_len_bytes + self.input_data +
            output_len_bytes + self.output_data + meta_bytes + level_bytes +
            error_bytes + self.prev_hash + op_id_bytes + ts_bytes + exec_time_bytes
        )

    def to_bytes(self) -> bytes:
        return self._to_bytes_without_signature() + self.signature

    def to_json(self) -> Dict[str, Any]:
        return {
            "step": str(self.step),
            "operation": self.operation,
            "input_data": base64.b64encode(self.input_data).decode('utf-8'),
            "output_data": base64.b64encode(self.output_data).decode('utf-8'),
            "metadata": self.metadata,
            "log_level": self.log_level,
            "error_code": self.error_code,
            "prev_hash": base64.b64encode(self.prev_hash).decode('utf-8'),
            "operation_id": self.operation_id,
            "timestamp": str(self.timestamp),
            "execution_time_ns": str(self.execution_time_ns),
            "signature": base64.b64encode(self.signature).decode('utf-8')
        }

class TCCLogger:
    def __init__(self):
        self.tcc_log: List[TCCLogEntry] = []
        self.step_counter: int = 0
        self.signing_key = nacl.signing.SigningKey.generate()
        self.verifying_key = self.signing_key.verify_key

    def log(self, operation: str, input_data: bytes, output_data: bytes,
            metadata: Dict[str, Any] = None, log_level: str = "INFO", error_code: str = "NONE") -> None:
        entry = TCCLogEntry(
            step=self.step_counter,
            operation=operation,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata or {},
            log_level=log_level,
            error_code=error_code,
            prev_hash=self._compute_prev_hash(),
            operation_id=hashlib.sha256(f"{self.step_counter}:{operation}:{time.time_ns()}".encode()).hexdigest()[:32],
            timestamp=time.time_ns(),
            execution_time_ns=0,
            signature=b''
        )
        entry_bytes = entry._to_bytes_without_signature()
        entry.signature = self.signing_key.sign(entry_bytes).signature
        entry.execution_time_ns = time.time_ns() - entry.timestamp
        self.tcc_log.append(entry)
        self.step_counter += 1

    def _compute_prev_hash(self) -> bytes:
        if not self.tcc_log:
            return b'\x00' * 32
        return hashlib.sha256(self.tcc_log[-1].to_bytes()).digest()

    def save_log(self, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8', errors='replace') as f:
            for entry in self.tcc_log:
                f.write(json.dumps(entry.to_json()) + '\n')

class BCIAdapter:
    def __init__(self, eeg_source):
        self.source = eeg_source

    def get_current_state(self) -> Dict[str, Any]:
        return self.source.read_brainwave_data()

    def interpret(self, state: Dict[str, Any]) -> Tuple[str, float]:
        alpha = state.get("alpha", 0.0)
        beta = state.get("beta", 0.0)
        if alpha > 0.7:
            return "relaxed", min(1.0, alpha)
        elif beta > 0.7:
            return "focused", min(1.0, beta)
        return "neutral", 0.5

class MockEEGSource:
    def read_brainwave_data(self) -> Dict[str, Any]:
        import random
        return {
            "alpha": random.uniform(0.0, 1.0),
            "beta": random.uniform(0.0, 1.0)
        }