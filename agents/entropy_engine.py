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
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

For Companies:
Commercial use by companies requires a separate license. Contact iconoclastdao@gmail.com
for licensing terms and conditions. Unauthorized commercial use is prohibited.
"""

import time
import hashlib
import random
from typing import Dict
from .tcc_logger import TCCLogger

class LLMEntropyEngine:
    FEE_PER_SAMPLING = 1000

    def __init__(self, initial_fee: int = 1000):
        self.fee_per_sampling = initial_fee
        self.commitments = {}
        self.commitment_timestamps = {}
        self.logger = TCCLogger()
        self.random_state = random.Random()

    def commit_sampling(self, user_id: str, seed: int, temperature: float) -> None:
        start_time = time.time_ns()
        try:
            commitment = hashlib.sha256(f"{seed}:{temperature}".encode()).digest()
            if user_id in self.commitments:
                raise ValueError("Commitment already exists for user")
            self.commitments[user_id] = (seed, temperature)
            self.commitment_timestamps[user_id] = time.time_ns()
            self.logger.log(
                "sampling_committed",
                commitment,
                b"",
                {"user_id": user_id, "seed": seed, "temperature": temperature, "execution_time_ns": time.time_ns() - start_time}
            )
        except Exception as e:
            self.logger.log(
                "sampling_committed",
                b"",
                b"",
                {"user_id": user_id, "error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "COMMIT_FAILED"
            )
            raise

    def reveal_sampling(self, user_id: str, seed: int, temperature: float, fee: int) -> None:
        start_time = time.time_ns()
        try:
            if fee < self.fee_per_sampling):
                raise ValueError("Insufficient fee")
            if user_id not in self.commitments:
                raise ValueError("No commitment found")
            if self.commitments[user_id] != (seed, temperature):
                raise ValueError("Invalid commitment")
            if (time.time_ns() - self.commitment_timestamps[user_id]) > 86400 * 1_000_000_000:
                raise ValueError("Commitment expired")
            del self.commitments[user_id]
            del self.commitment_timestamps[user_id]
            self.random_state.seed(seed)
            self.logger.log(
                "sampling_revealed",
                f"{seed}:{temperature}".encode(),
                b"",
                {"user_id": user_id, "execution_time_ns": time.time_ns() - start_time}
            )
        except Exception as e:
            self.logger.log(
                "sampling_revealed",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "REVEAL_FAILED"
            )
            raise

    def save_log(self, filename: str) -> None:
        self.logger.save_log(filename)

class LLMCoordinator:
    def __init__(self, engine_a: LLMEntropyEngine, engine_b: LLMEntropyEngine, engine_c: LLMEntropyEngine):
        if not all([engine_a, engine_b, engine_c]) or len(set([id(engine_a), id(engine_b), id(engine_c)])) != 3:
            raise ValueError("Invalid or duplicate engines")
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.engine_c = engine_c
        self.logger = TCCLogger()

    def commit_sampling_all(self, user_id: str, seed_a: int, temp_a: float, seed_b: int, temp_b: float, seed_c: int, temp_c: float) -> None:
        start_time = time.time_ns()
        try:
            self.engine_a.commit_sampling(user_id, seed_a, temp_a)
            self.engine_b.commit_sampling(user_id, seed_b, temp_b)
            self.engine_c.commit_sampling(user_id, seed_c, temp_c)
            self.logger.log(
                "commit_sampling_all",
                f"{seed_a}:{temp_a}:{seed_b}:{temp_b}:{seed_c}:{temp_c}".encode(),
                b"",
                {"user_id": user_id, "execution_time_ns": time.time_ns() - start_time}
            )
        except Exception as e:
            self.logger.log(
                "commit_sampling_all",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "COMMIT_ALL_FAILED"
            )
            raise

    def reveal_sampling_all(self, user_id: str, seed_a: int, temp_a: float, seed_b: int, temp_b: float, seed_c: int, temp_c: float, fee: int) -> None:
        start_time = time.time_ns()
        try:
            total_fee = 0
            has_a = user_id in self.engine_a.commitments
            has_b = user_id in self.engine_b.commitments
            has_c = user_id in self.engine_c.commitments
            if has_a:
                total_fee += self.engine_a.fee_per_sampling
            if has_b:
                total_fee += self.engine_b.fee_per_sampling
            if has_c:
                total_fee += self.engine_c.fee_per_sampling
            if fee < total_fee:
                raise ValueError("Insufficient fee")
            if has_a:
                self.engine_a.reveal_sampling(user_id, seed_a, temp_a, self.engine_a.fee_per_sampling)
            if has_b:
                self.engine_b.reveal_sampling(user_id, seed_b, temp_b, self.engine_b.fee_per_sampling)
            if has_c:
                self.engine_c.reveal_sampling(user_id, seed_c, temp_c, self.engine_c.fee_per_sampling)
            self.logger.log(
                "reveal_sampling_all",
                f"{seed_a}:{temp_a}:{seed_b}:{temp_b}:{seed_c}:{temp_c}".encode(),
                b"",
                {"user_id": user_id, "total_fee": total_fee, "execution_time_ns": time.time_ns() - start_time}
            )
        except Exception as e:
            self.logger.log(
                "reveal_sampling_all",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "REVEAL_ALL_FAILED"
            )
            raise

    def save_log(self, filename: str) -> None:
        self.logger.save_log(filename)