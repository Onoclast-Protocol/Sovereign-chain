"""
Dual License

For Open-Source Individuals:
Divine Covenant License (DCL-1.0)

Copyright (c) 2025 James B. Chapman, IconoclastDAO

Permission is hereby granted, free of charge, to any individual obtaining a copy
of this software and associated documentation files (the "Software"), for personal,
non-commercial use, provided the use aligns with truth, sovereignty, and non-coercion.
Derivatives must honor the origin, uphold transparency, and ensure evolution.
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

import json
import time
import hashlib
import base64
import struct
import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any, Tuple
import nacl.signing
import nacl.encoding
import nacl.exceptions
from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from .security import BioInspiredSecuritySystem
from .soulbound_identity import SoulBoundIdentitySystem
from .yield_protocol import YieldProtocol
from .messaging import Messaging
from .consensus import Consensus
from .voucher_system import VoucherSystem
from .town_square import TownSquare
from ..agents.tcc_logger import TCCLogger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("onoclast_pulse")

# Environment variables
ARBITRUM_RPC_URL = "https://arb1.arbitrum.io/rpc"
PULSE_CONTRACT_ADDRESS = Web3.to_checksum_address("0xYourPulseContractAddress")  # Placeholder
ADMIN_PRIVATE_KEY = "0xYourAdminPrivateKey"  # Placeholder, store securely

# Pulse Contract ABI (simplified)
PULSE_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "pulseName", "type": "string"},
            {"internalType": "address", "name": "creator", "type": "address"},
            {"internalType": "bytes", "name": "configCid", "type": "bytes"},
        ],
        "name": "createPulse",
        "outputs": [{"internalType": "uint256", "name": "pulseId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "string", "name": "pulseName", "type": "string"},
            {"internalType": "string", "name": "signalName", "type": "string"},
            {"internalType": "address", "name": "fromAgent", "type": "address"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "strength", "type": "uint256"},
        ],
        "name": "emitSignal",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

# === Mock EEG Source ===
class MockEEGSource:
    def read_brainwave_data(self) -> Dict[str, Any]:
        return {
            "alpha": random.uniform(0.0, 1.0),
            "beta": random.uniform(0.0, 1.0),
            "theta": random.uniform(0.0, 1.0),
        }

# === BCI Adapter ===
class BCIAdapter:
    def __init__(self, eeg_source):
        self.source = eeg_source

    def get_current_state(self) -> Dict[str, Any]:
        return self.source.read_brainwave_data()

    def interpret(self, state: Dict[str, Any]) -> Tuple[str, float]:
        alpha = state.get("alpha", 0.0)
        beta = state.get("beta", 0.0)
        theta = state.get("theta", 0.0)
        if alpha > 0.7:
            return "relaxed", min(1.0, alpha)
        elif beta > 0.7:
            return "focused", min(1.0, beta)
        elif theta > 0.7:
            return "meditative", min(1.0, theta)
        return "neutral", 0.5

# === LLM Module for Self-Evolving AI ===
class ModelManager:
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

class IntentProcessor:
    def __init__(self, model_manager: ModelManager):
        self.tokenizer = model_manager.tokenizer
        self.model = model_manager.model
        self.device = model_manager.device
        self.logger = TCCLogger()

    def process_intent(self, input_text: str, context: Dict[str, Any]) -> str:
        start_time = time.time_ns()
        prompt = f"Context: {json.dumps(context)}\nIntent: {input_text}\nAction: "
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50,
            temperature=0.7,
            top_k=50,
            do_sample=True
        )
        action = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Action: ")[-1].strip()
        self.logger.log(
            "intent_process",
            prompt.encode('utf-8'),
            action.encode('utf-8'),
            {"input_length": len(inputs["input_ids"][0]), "execution_time_ns": time.time_ns() - start_time},
            "INFO",
            "SUCCESS"
        )
        return action

# === Transparency Ledger ===
@dataclass
class LedgerEntry:
    timestamp: float
    pulse_name: str
    operation: str
    details: Dict[str, Any]
    signature: bytes
    description: str
    verifying_key: Optional[bytes] = None

class TransparencyLedger:
    def __init__(self):
        self.entries: List[LedgerEntry] = []
        self.logger = TCCLogger()
        self.signing_key = nacl.signing.SigningKey.generate()
        self.verifying_key = self.signing_key.verify_key

    def add_entry(self, pulse_name: str, operation: str, details: Dict[str, Any], description: str) -> None:
        timestamp = time.time()
        details_bytes = json.dumps(details, sort_keys=True).encode('utf-8')
        signature = self.signing_key.sign(details_bytes).signature
        entry = LedgerEntry(
            timestamp=timestamp,
            pulse_name=pulse_name,
            operation=operation,
            details=details,
            signature=signature,
            description=description,
            verifying_key=self.verifying_key.encode(encoder=nacl.encoding.RawEncoder)
        )
        self.entries.append(entry)
        self.logger.log(
            "ledger_entry",
            details_bytes,
            signature,
            {"pulse_name": pulse_name, "operation": operation, "timestamp": timestamp},
            "INFO",
            "SUCCESS"
        )

    def verify_entry(self, entry: LedgerEntry) -> bool:
        if not entry.verifying_key or len(entry.verifying_key) != 32:
            logger.info(f"Invalid entry skipped for {entry.pulse_name} ({entry.operation})")
            return False
        try:
            details_bytes = json.dumps(entry.details, sort_keys=True).encode('utf-8')
            verifying_key = nacl.signing.VerifyKey(entry.verifying_key)
            verifying_key.verify(details_bytes, entry.signature)
            return True
        except nacl.exceptions.BadSignatureError:
            logger.warning(f"Signature verification failed for entry {entry.pulse_name} ({entry.operation})")
            return False

    def save_ledger(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8', errors='replace') as f:
            for entry in self.entries:
                f.write(json.dumps({
                    "timestamp": entry.timestamp,
                    "pulse_name": entry.pulse_name,
                    "operation": entry.operation,
                    "details": entry.details,
                    "signature": base64.b64encode(entry.signature).decode('utf-8'),
                    "description": entry.description,
                    "verifying_key": base64.b64encode(entry.verifying_key).decode('utf-8') if entry.verifying_key else ""
                }) + '\n')

    def load_ledger(self, filename: str, clear_if_invalid: bool = False) -> None:
        if not os.path.exists(filename):
            return
        invalid_detected = False
        self.entries.clear()
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    verifying_key = base64.b64decode(data["verifying_key"]) if data.get("verifying_key") else None
                    if verifying_key and len(verifying_key) != 32:
                        logger.info(f"Skipping entry {data.get('pulse_name', 'unknown')} due to invalid verifying key")
                        invalid_detected = True
                        continue
                    entry = LedgerEntry(
                        timestamp=data["timestamp"],
                        pulse_name=data["pulse_name"],
                        operation=data["operation"],
                        details=data["details"],
                        signature=base64.b64decode(data["signature"]),
                        description=data.get("description", "No description"),
                        verifying_key=verifying_key
                    )
                    self.entries.append(entry)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Failed to load ledger entry: {e}")
                    invalid_detected = True
        if clear_if_invalid and invalid_detected:
            logger.info(f"Invalid entries detected; clearing {filename}")
            self.entries.clear()
            if os.path.exists(filename):
                os.remove(filename)

# === Pulse Definitions ===
@dataclass
class Signal:
    from_agent: str
    name: str
    time: float
    strength: float = 1.0
    signature: Optional[bytes] = None

@dataclass
class Action:
    type: str
    signal: Optional[str] = None
    to: Optional[Union[str, List[str]]] = None
    phase_shift: Optional[float] = None
    fraction: Optional[int] = None
    condition: Optional[str] = None
    strength: Optional[float] = None
    intent: Optional[str] = None

@dataclass
class Pulse:
    name: str
    interval: float
    next_fire: float
    body: List[Action]
    time_scale: float = 1.0
    fractions: int = 1
    enabled: bool = True
    inbox: List[Signal] = field(default_factory=list)
    logger: TCCLogger = field(default_factory=TCCLogger)
    signing_key: nacl.signing.SigningKey = field(default_factory=nacl.signing.SigningKey.generate)
    bci_adapter: Optional[BCIAdapter] = None
    intent_processor: Optional[IntentProcessor] = None

    def should_fire(self, global_time: float) -> bool:
        if not self.enabled:
            self.logger.log(
                "pulse_check",
                str(global_time).encode('utf-8'),
                b"disabled",
                {"pulse_name": self.name, "enabled": self.enabled},
                "INFO",
                "PULSE_DISABLED"
            )
            return False
        local_time = global_time * self.time_scale
        should_fire = abs(local_time - self.next_fire) < 1e-6
        self.logger.log(
            "pulse_check",
            str(global_time).encode('utf-8'),
            b"firing" if should_fire else b"not_ready",
            {"pulse_name": self.name, "global_time": global_time, "local_time": local_time, "next_fire": self.next_fire},
            "INFO",
            "PULSE_FIRING" if should_fire else "PULSE_NOT_READY"
        )
        return should_fire

    def on_signal(self, signal: Signal, state: 'PulseState', ledger: TransparencyLedger) -> None:
        if signal.signature:
            try:
                verifying_key = nacl.signing.VerifyKey(self.signing_key.verify_key.encode(encoder=nacl.encoding.RawEncoder))
                verifying_key.verify(
                    f"{signal.from_agent}:{signal.name}:{signal.time}:{signal.strength}".encode('utf-8'),
                    signal.signature
                )
            except nacl.exceptions.BadSignatureError:
                self.logger.log(
                    "signal_verification",
                    signal.name.encode('utf-8'),
                    b"failed",
                    {"from_agent": signal.from_agent, "signal_time": signal.time, "strength": signal.strength},
                    "ERROR",
                    "INVALID_SIGNATURE"
                )
                return
        description = f"Received signal {signal.name} from {signal.from_agent} at time {signal.time} with strength {signal.strength}"
        self.logger.log(
            "signal_received",
            signal.name.encode('utf-8'),
            b"processed",
            {"from_agent": signal.from_agent, "signal_name": signal.name, "signal_time": signal.time, "strength": signal.strength},
            "INFO",
            "SUCCESS"
        )
        ledger.add_entry(self.name, "signal_received", {
            "from_agent": signal.from_agent,
            "signal_name": signal.name,
            "signal_time": signal.time,
            "strength": signal.strength
        }, description)
        if signal.name == "sync":
            self.next_fire += signal.strength
            self.logger.log(
                "signal_sync",
                signal.name.encode('utf-8'),
                b"modulated",
                {"pulse_name": self.name, "next_fire": self.next_fire, "strength": signal.strength},
                "INFO",
                "SYNC_APPLIED"
            )
        elif signal.name == "off":
            self.enabled = False
            self.logger.log(
                "signal_off",
                signal.name.encode('utf-8'),
                b"disabled",
                {"pulse_name": self.name, "enabled": self.enabled},
                "INFO",
                "PULSE_DISABLED"
            )
        elif signal.name == "on":
            self.enabled = True
            self.logger.log(
                "signal_on",
                signal.name.encode('utf-8'),
                b"enabled",
                {"pulse_name": self.name, "enabled": self.enabled},
                "INFO",
                "PULSE_ENABLED"
            )

    def fire(self, global_time: float, state: 'PulseState', ledger: TransparencyLedger) -> List[Signal]:
        if not self.should_fire(global_time):
            return []
        if self.fractions <= 0:
            self.logger.log(
                "pulse_error",
                str(global_time).encode('utf-8'),
                b"invalid fractions",
                {"pulse_name": self.name, "fractions": self.fractions},
                "ERROR",
                "INVALID_FRACTIONS"
            )
            return []
        self.logger.log(
            "pulse_fire",
            str(global_time).encode('utf-8'),
            b"fired",
            {"pulse_name": self.name, "global_time": global_time},
            "INFO",
            "PULSE_FIRED"
        )
        signals_emitted = []
        local_time = global_time * self.time_scale
        self.next_fire += self.interval
        for f in range(self.fractions):
            for action in self.body:
                if action.fraction is not None and action.fraction != f:
                    continue
                action_strength = min(1.0, max(0.0, action.strength if action.strength is not None else 1.0))
                if action.condition:
                    condition_met, condition_strength = self.evaluate_condition(action.condition)
                    if not condition_met:
                        self.logger.log(
                            "action_skip",
                            action.signal.encode('utf-8') if action.signal else b'',
                            b"condition not met",
                            {"pulse_name": self.name, "condition": action.condition, "strength": action_strength},
                            "INFO",
                            "CONDITION_FAIL"
                        )
                        continue
                    action_strength *= condition_strength
                if action.intent and self.intent_processor:
                    context = {"pulse_name": self.name, "action": action.type, "signal": action.signal}
                    action_signal = self.intent_processor.process_intent(action.intent, context)
                else:
                    action_signal = action.signal
                action_details = {
                    "pulse_name": self.name,
                    "action_type": action.type,
                    "signal": action_signal,
                    "to": action.to,
                    "phase_shift": action.phase_shift,
                    "fraction": action.fraction,
                    "condition": action.condition,
                    "strength": action_strength,
                    "intent": action.intent
                }
                description = f"Performed action: {action.type} signal {action_signal} to {action.to} at time {global_time} with strength {action_strength}"
                ledger.add_entry(self.name, action.type, action_details, description)
                if action.type == "emit" and action_signal:
                    targets = [action.to] if isinstance(action.to, str) else (action.to or [])
                    for target in targets:
                        sig = Signal(
                            from_agent=self.name,
                            name=action_signal,
                            time=global_time,
                            strength=action_strength,
                            signature=self.sign_signal(action_signal, global_time, action_strength)
                        )
                        if target in state.pulses:
                            state.pulses[target].inbox.append(sig)
                            self.logger.log(
                                "action_emit",
                                action_signal.encode('utf-8'),
                                b"emitted",
                                {"target": target, "signal_name": action_signal, "strength": action_strength},
                                "INFO",
                                "ACTION_EMITTED"
                            )
                        signals_emitted.append(sig)
                        state.signals.append(sig)
                elif action.type == "broadcast" and action_signal:
                    for target in state.pulses:
                        if target != self.name:
                            sig = Signal(
                                from_agent=self.name,
                                name=action_signal,
                                time=global_time,
                                strength=action_strength,
                                signature=self.sign_signal(action_signal, global_time, action_strength)
                            )
                            state.pulses[target].inbox.append(sig)
                            self.logger.log(
                                "action_broadcast",
                                action_signal.encode('utf-8'),
                                b"broadcasted",
                                {"target": target, "signal_name": action_signal, "strength": action_strength},
                                "INFO",
                                "ACTION_BROADCASTED"
                            )
                            signals_emitted.append(sig)
                            state.signals.append(sig)
                elif action.type == "modulate" and action.phase_shift is not None:
                    if action.to in state.pulses:
                        target_pulse = state.pulses[action.to]
                        target_pulse.next_fire += action.phase_shift * action_strength
                        self.logger.log(
                            "action_modulate",
                            action_signal.encode('utf-8') if action_signal else b'',
                            b"modulated",
                            {"target": action.to, "phase_shift": action.phase_shift, "strength": action_strength},
                            "INFO",
                            "ACTION_MODULATED"
                        )
                elif action.type == "modulate_time" and action.fraction is not None:
                    if action.fraction == 0:
                        self.logger.log(
                            "action_modulate_time_error",
                            str(self.time_scale).encode('utf-8'),
                            b"invalid fraction",
                            {"pulse_name": self.name, "fraction": action.fraction},
                            "ERROR",
                            "INVALID_FRACTION"
                        )
                        continue
                    old_time_scale = self.time_scale
                    self.time_scale = max(0.1, min(self.time_scale / action.fraction * action_strength, 10.0))
                    self.logger.log(
                        "action_modulate_time",
                        str(old_time_scale).encode('utf-8'),
                        str(self.time_scale).encode('utf-8'),
                        {"pulse_name": self.name, "fraction": action.fraction, "new_time_scale": self.time_scale},
                        "INFO",
                        "ACTION_MODULATED_TIME"
                    )
        for signal in self.inbox:
            self.on_signal(signal, state, ledger)
        self.inbox.clear()
        return signals_emitted

    def evaluate_condition(self, condition: str) -> Tuple[bool, float]:
        if self.bci_adapter:
            brain_state, strength = self.bci_adapter.interpret(self.bci_adapter.get_current_state())
            return brain_state == condition, strength
        return True, 1.0

    def sign_signal(self, signal_name: str, timestamp: float, strength: float) -> bytes:
        msg = f"{self.name}:{signal_name}:{timestamp}:{strength}".encode('utf-8')
        return self.signing_key.sign(msg).signature

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "interval": self.interval,
            "next_fire": self.next_fire,
            "body": [
                {
                    "type": a.type,
                    "signal": a.signal,
                    "to": a.to,
                    "phase_shift": a.phase_shift,
                    "fraction": a.fraction,
                    "condition": a.condition,
                    "strength": a.strength,
                    "intent": a.intent
                } for a in self.body
            ],
            "time_scale": self.time_scale,
            "fractions": self.fractions,
            "enabled": self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], bci_adapter: Optional[BCIAdapter] = None, intent_processor: Optional[IntentProcessor] = None) -> 'Pulse':
        return cls(
            name=data["name"],
            interval=data["interval"],
            next_fire=data["next_fire"],
            body=[Action(**action) for action in data["body"]],
            time_scale=data.get("time_scale", 1.0),
            fractions=data.get("fractions", 1),
            enabled=data.get("enabled", True),
            bci_adapter=bci_adapter,
            intent_processor=intent_processor
        )

class QuantumPulse(Pulse):
    def __init__(self, *args, state_vector: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_vector = state_vector or {"on": 0.5, "off": 0.5}

    def should_fire(self, global_time: float) -> bool:
        if not self.enabled:
            self.logger.log(
                "pulse_check",
                str(global_time).encode('utf-8'),
                b"disabled",
                {"pulse_name": self.name, "enabled": self.enabled},
                "INFO",
                "PULSE_DISABLED"
            )
            return False
        collapsed_state, strength = self.collapse_state()
        should_fire = collapsed_state == "on"
        self.logger.log(
            "pulse_check",
            str(global_time).encode('utf-8'),
            b"firing" if should_fire else b"not_firing",
            {"pulse_name": self.name, "global_time": global_time, "state": collapsed_state, "strength": strength},
            "INFO",
            "PULSE_FIRING" if should_fire else "PULSE_NOT_READY"
        )
        return should_fire

    def collapse_state(self) -> Tuple[str, float]:
        rand = random.random()
        cumulative = 0.0
        for state, prob in self.state_vector.items():
            cumulative += prob
            if rand <= cumulative:
                strength = prob if state == "on" else 1.0 - prob
                return state, min(1.0, max(0.0, strength))
        return "off", 0.5

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["is_quantum"] = True
        data["state_vector"] = self.state_vector
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], bci_adapter: Optional[BCIAdapter] = None, intent_processor: Optional[IntentProcessor] = None) -> 'QuantumPulse':
        return cls(
            name=data["name"],
            interval=data["interval"],
            next_fire=data["next_fire"],
            body=[Action(**action) for action in data["body"]],
            time_scale=data.get("time_scale", 1.0),
            fractions=data.get("fractions", 1),
            enabled=data.get("enabled", True),
            state_vector=data.get("state_vector", {"on": 0.5, "off": 0.5}),
            bci_adapter=bci_adapter,
            intent_processor=intent_processor
        )

# === Pulse State ===
@dataclass
class PulseState:
    time: float = 0.0
    pulses: Dict[str, Union[Pulse, QuantumPulse]] = field(default_factory=dict)
    signals: List[Signal] = field(default_factory=list)

    def save_state(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        state_data = {
            "time": self.time,
            "pulses": {name: pulse.to_dict() for name, pulse in self.pulses.items()},
            "signals": [
                {
                    "from_agent": s.from_agent,
                    "name": s.name,
                    "time": s.time,
                    "strength": s.strength,
                    "signature": base64.b64encode(s.signature).decode('utf-8') if s.signature else None
                } for s in self.signals
            ]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2)

    def load_state(self, filename: str, bci_adapter: Optional[BCIAdapter] = None, intent_processor: Optional[IntentProcessor] = None) -> None:
        if not os.path.exists(filename):
            return
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.time = data.get("time", 0.0)
            self.pulses = {}
            for name, pulse_data in data.get("pulses", {}).items():
                if pulse_data.get("is_quantum", False):
                    self.pulses[name] = QuantumPulse.from_dict(pulse_data, bci_adapter, intent_processor)
                else:
                    self.pulses[name] = Pulse.from_dict(pulse_data, bci_adapter, intent_processor)
            self.signals = [
                Signal(
                    from_agent=s["from_agent"],
                    name=s["name"],
                    time=s["time"],
                    strength=s.get("strength", 1.0),
                    signature=base64.b64decode(s["signature"]) if s["signature"] else None
                ) for s in data.get("signals", [])
            ]

# === Pulse System ===
class PulseSystem:
    KARMA_PER_PULSE_CREATION = 200
    KARMA_PER_SIGNAL = 50
    MIN_KARMA_TO_CREATE_PULSE = 500

    def __init__(
        self,
        soulbound_system: SoulBoundIdentitySystem,
        yield_protocol: YieldProtocol,
        messaging: Messaging,
        consensus: Consensus,
        voucher_system: VoucherSystem,
        town_square: TownSquare,
        security: BioInspiredSecuritySystem,
        model_name: str = "distilgpt2",
        state_file: str = "data/pulse_state.json",
        ledger_file: str = "data/pulse_ledger.json",
        log_dir: str = "data/pulse_logs/"
    ):
        self.logger = TCCLogger()
        self.security = security
        self.soulbound_system = soulbound_system
        self.yield_protocol = yield_protocol
        self.messaging = messaging
        self.consensus = consensus
        self.voucher_system = voucher_system
        self.town_square = town_square
        self.state = PulseState()
        self.ledger = TransparencyLedger()
        self.w3 = Web3(Web3.HTTPProvider(ARBITRUM_RPC_URL, request_kwargs={"timeout": 30}))
        if not self.w3.is_connected():
            raise RuntimeError("Web3 connection failed")
        self.contract = self.w3.eth.contract(address=PULSE_CONTRACT_ADDRESS, abi=PULSE_ABI)
        self.admin_account = Account.from_key(ADMIN_PRIVATE_KEY)
        self.model_manager = ModelManager(model_name)
        self.intent_processor = IntentProcessor(self.model_manager)
        self.bci_adapter = BCIAdapter(MockEEGSource())
        self.state_file = state_file
        self.ledger_file = ledger_file
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.state.load_state(state_file, self.bci_adapter, self.intent_processor)
        self.ledger.load_ledger(ledger_file, clear_if_invalid=True)

    def _sign_transaction(self, tx: Dict) -> Dict:
        try:
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.admin_account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if receipt.status != 1:
                raise RuntimeError("Transaction failed")
            return receipt
        except Exception as e:
            self.logger.log(
                "sign_transaction",
                json.dumps(tx).encode(),
                b"",
                {"error": str(e)},
                "ERROR",
                "SIGN_TX_FAILED"
            )
            raise

    def create_pulse(
        self,
        creator_address: str,
        pulse_config: Dict[str, Any],
        signature: str,
        voucher_id: Optional[int] = None
    ) -> str:
        start_time = time.time_ns()
        try:
            # Verify identity and karma
            if not self.soulbound_system.verify_identity(creator_address):
                raise ValueError("Invalid creator identity")
            karma = self.yield_protocol.get_karma(creator_address)
            if karma < self.MIN_KARMA_TO_CREATE_PULSE:
                raise ValueError(f"Creator karma {karma} below required {self.MIN_KARMA_TO_CREATE_PULSE}")
            # Verify signature
            config_str = json.dumps(pulse_config, sort_keys=True)
            message = encode_defunct(text=f"create_pulse:{config_str}:{voucher_id or 0}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != creator_address.lower():
                raise ValueError("Invalid signature")
            # Security check
            is_anomaly, score, explanation = self.security.detect_anomaly(config_str.encode())
            if is_anomaly:
                self.logger.log(
                    "create_pulse_anomaly",
                    config_str.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")
            # Verify voucher if provided
            if voucher_id:
                voucher = self.voucher_system.get_voucher(voucher_id)
                if voucher["redeemed"]:
                    raise ValueError("Voucher already redeemed")
                if voucher["depositor"] != creator_address:
                    raise ValueError("Voucher not owned by creator")
            # Create pulse
            pulse_name = pulse_config["name"]
            if pulse_name in self.state.pulses:
                raise ValueError("Pulse name already exists")
            # Log on-chain
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.createPulse(pulse_name, creator_address, config_str.encode()).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 300000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            receipt = self._sign_transaction(tx)
            # Instantiate pulse
            if pulse_config.get("is_quantum", False):
                pulse = QuantumPulse.from_dict(pulse_config, self.bci_adapter, self.intent_processor)
            else:
                pulse = Pulse.from_dict(pulse_config, self.bci_adapter, self.intent_processor)
            self.state.pulses[pulse_name] = pulse
            self.state.save_state(self.state_file)
            # Log action
            self.ledger.add_entry(
                pulse_name,
                "create_pulse",
                {"creator": creator_address, "config": pulse_config, "voucher_id": voucher_id or 0},
                f"Created pulse {pulse_name} by {creator_address}"
            )
            self.ledger.save_ledger(self.ledger_file)
            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=creator_address,
                amount=self.KARMA_PER_PULSE_CREATION,
                action_type="pulse_creation",
                signature=signature
            )
            self.logger.log(
                "create_pulse",
                config_str.encode(),
                pulse_name.encode(),
                {
                    "creator": creator_address,
                    "pulse_name": pulse_name,
                    "voucher_id": voucher_id or 0,
                    "karma_earned": self.KARMA_PER_PULSE_CREATION,
                    "execution_time_ns": time.time_ns() - start_time
                },
                "INFO",
                "SUCCESS"
            )
            return pulse_name
        except Exception as e:
            self.logger.log(
                "create_pulse",
                config_str.encode() if 'config_str' in locals() else b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "CREATE_PULSE_FAILED"
            )
            raise

    def process_signals(self, global_time: float) -> List[Signal]:
        start_time = time.time_ns()
        signals_emitted = []
        for pulse_name in list(self.state.pulses.keys()):
            pulse = self.state.pulses[pulse_name]
            new_signals = pulse.fire(global_time, self.state, self.ledger)
            for sig in new_signals:
                self.ledger.add_entry(
                    pulse_name,
                    "fire",
                    {"signal": sig.name, "time": sig.time, "strength": sig.strength},
                    f"Signal {sig.name} emitted from {pulse_name} at time {sig.time}"
                )
                # Handle protocol-specific signals
                if sig.name == "create_post" and isinstance(pulse, Pulse):
                    context = {"pulse_name": pulse_name, "signal": sig.name, "strength": sig.strength}
                    content = self.intent_processor.process_intent("Generate a community post", context)
                    try:
                        post_id = self.town_square.create_post(
                            content=content,
                            author_address=pulse.signing_key.verify_key.encode().hex(),
                            voucher_id=None,
                            signature=pulse.sign_signal("create_post", global_time, sig.strength).hex()
                        )
                        self.messaging.broadcast_message(
                            f"New post {post_id} created by pulse {pulse_name}: {content[:50]}...",
                            pulse.signing_key.verify_key.encode().hex()
                        )
                    except Exception as e:
                        self.logger.log(
                            "signal_post",
                            sig.name.encode(),
                            b"",
                            {"error": str(e)},
                            "ERROR",
                            "POST_CREATION_FAILED"
                        )
                signals_emitted.extend(new_signals)
        self.state.time = global_time
        self.state.save_state(self.state_file)
        self.ledger.save_ledger(self.ledger_file)
        for pulse in self.state.pulses.values():
            pulse.logger.save_log(f"{self.log_dir}{pulse.name}_log.json")
        self.logger.log(
            "process_signals",
            str(global_time).encode(),
            json.dumps([s.__dict__ for s in signals_emitted]).encode(),
            {"pulse_count": len(self.state.pulses), "signal_count": len(signals_emitted)},
            "INFO",
            "SUCCESS"
        )
        return signals_emitted

    def get_pulse(self, pulse_name: str) -> Dict[str, Any]:
        if pulse_name not in self.state.pulses:
            raise ValueError("Pulse not found")
        return self.state.pulses[pulse_name].to_dict()

    def get_ledger_entries(self, pulse_name: Optional[str] = None) -> List[Dict[str, Any]]:
        entries = [
            {
                "timestamp": e.timestamp,
                "pulse_name": e.pulse_name,
                "operation": e.operation,
                "details": e.details,
                "description": e.description,
                "verified": self.ledger.verify_entry(e)
            } for e in self.ledger.entries if pulse_name is None or e.pulse_name == pulse_name
        ]
        self.logger.log(
            "get_ledger",
            pulse_name.encode() if pulse_name else b"",
            json.dumps(entries).encode(),
            {"entry_count": len(entries)},
            "INFO",
            "SUCCESS"
        )
        return entries