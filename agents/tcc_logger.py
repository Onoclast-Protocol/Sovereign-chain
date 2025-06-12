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
import gzip
import json
import struct
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
import logging

logger = logging.getLogger("onoclast_protocol")

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
    timestamp: int = field(default_factory=lambda: time.time_ns())
    execution_time_ns: int = 0
    signing_key: bytes = field(default_factory=lambda: os.urandom(32))

    def _to_bytes(self):
        meta = json.dumps(self.metadata).encode()
        return b"".join([
            struct.pack(">Q", self.timestamp),
            struct.pack(">I", self.step),
            self.operation.encode().ljust(32, b'\0')[:32],
            self.input_data,
            self.output_data,
            meta,
            self.prev_hash
        ])

    def to_bytes(self):
        b = self._to_bytes()
        sig = hashlib.blake2b(b + self.signing_key, digest_size=32).digest()
        return gzip.compress(b + sig)

class TCCLogger:
    def __init__(self, level: str = "INFO"):
        self.level = getattr(logging, level)
        self.logs: List[TCCLogEntry] = []
        self.step = 0
        self.signing_key = os.urandom(32)

    def _prev_hash(self) -> bytes:
        return hashlib.sha256(self.logs[-1].to_bytes()).digest() if self.logs else b'\x00' * 32

    def log(self, op: str, inp: bytes, outp: bytes, metadata: Dict[str, Any] = None, lvl: str = "INFO", err: str = "NONE"):
        if getattr(logging, lvl) < self.level:
            return
        entry = TCCLogEntry(
            step=self.step,
            operation=op,
            input_data=inp,
            output_data=outp,
            metadata=metadata or {},
            log_level=lvl,
            error_code=err,
            prev_hash=self._prev_hash(),
            signing_key=self.signing_key
        )
        self.logs.append(entry)
        logger.log(self.level, f"[{op}] step={self.step} err={err}")
        self.step += 1

    def export_logs(self) -> List[Dict[str, Any]]:
        return [{
            "step": entry.step,
            "operation": entry.operation,
            "log_level": entry.log_level,
            "error_code": entry.error_code,
            "metadata": entry.metadata,
            "timestamp": entry.timestamp
        } for entry in self.logs]