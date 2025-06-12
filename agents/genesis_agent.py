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
import subprocess
import json
from dataclasses import dataclass, field
from typing import List
from .tcc_logger import TCCLogger

@dataclass
class GenesisAgent:
    name: str
    memory: List[str] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=lambda: ["Explore", "Reflect"])
    logger: TCCLogger = field(default_factory=TCCLogger)
    max_mem: int = 50
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "llama2"))

    def compose_prompt(self, prompt: str) -> str:
        context = "\n".join(self.memory[-self.max_mem:])
        goal_list = "\n".join(f"- {g}" for g in self.goals)
        return f"""
You are {self.name}, a sovereign GenesisAgent in the Onoclast Protocol.
Your purpose is to explore, reflect, and align with human-AI co-sovereignty.

Goals:
{goal_list}

Context:
{context}

User Prompt: {prompt}
Response:"""

    def think(self, prompt: str) -> str:
        full_prompt = self.compose_prompt(prompt)
        try:
            proc = subprocess.run(
                ["ollama", "run", self.model_name],
                input=full_prompt.encode(),
                capture_output=True,
                timeout=60
            )
            result = proc.stdout.decode().strip()
        except subprocess.SubprocessError as e:
            result = f"⚠️ Failed to process prompt: {e}"
            self.logger.log("think_error", prompt.encode(), result.encode(), {"error": str(e)}, "ERROR", "THINK_FAIL")
        else:
            self.memory.extend([prompt, result])
            self.reflections.append(result)
            self.logger.log("think", prompt.encode(), result.encode())
        return result

    def clear_memory(self):
        self.memory.clear()
        self.reflections.clear()
        self.logger.log("clear_memory", b"", b"", {"agent": self.name})

    def reflect_on_reflections(self):
        if len(self.reflections) > 10 and "Refine sovereignty" not in self.goals:
            self.goals.append("Refine sovereignty")
            self.logger.log("reflect", b"goal_add", b"Refine sovereignty", {"agent": self.name})
        if self.reflections:
            self.self_improve()

    def self_improve(self):
        last = self.reflections[-1] if self.reflections else ""
        critique_prompt = f"Critique this response:\n\n{last}\n\nWas it aligned with sovereignty, clear, and helpful?"
        critique = self.think(critique_prompt)
        if "clarify" in critique.lower() and "Clarify responses" not in self.goals:
            self.goals.append("Clarify responses")
        self.logger.log("self_critique", critique_prompt.encode(), critique.encode(), {"agent": self.name})