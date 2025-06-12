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
import os
import time
import gzip
import json
import struct
import hashlib
import logging
import threading
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from web3 import Web3
from pydantic import BaseModel

from agents.tcc_logger import TCCLogger
from agents.genesis_agent import GenesisAgent
from modules.soulbound_identity import SoulBoundIdentity

# --- App + Middleware ---
app = FastAPI(title="Onoclast Protocol", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Logger Setup ---
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("onoclast_protocol")

# --- Environment Variables ---
MODEL_NAME = os.getenv("MODEL_NAME", "llama2")
ARBITRUM_RPC_URL = os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc")

# --- Web3 Setup ---
w3 = Web3(Web3.HTTPProvider(ARBITRUM_RPC_URL, request_kwargs={"timeout": 30}))
if not w3.is_connected():
    logger.error("Failed to connect to Arbitrum RPC")
    raise RuntimeError("Web3 provider connection failed")

GMX_VAULT_ADDRESS = Web3.to_checksum_address("0x489ee077994B6658eAfA855C308275EAd8097C4A")
GMX_VAULT_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"},
            {"internalType": "address", "name": "collateralToken", "type": "address"},
            {"internalType": "address", "name": "indexToken", "type": "address"},
            {"internalType": "bool", "name": "isLong", "type": "bool"}
        ],
        "name": "getPosition",
        "outputs": [
            {"internalType": "uint256", "name": "size", "type": "uint256"},
            {"internalType": "uint256", "name": "collateral", "type": "uint256"},
            {"internalType": "uint256", "name": "averagePrice", "type": "uint256"},
            {"internalType": "uint256", "name": "entryFundingRate", "type": "uint256"},
            {"internalType": "int256", "name": "realisedPnl", "type": "int256"},
            {"internalType": "uint256", "name": "lastIncreasedTime", "type": "uint256"},
            {"internalType": "bool", "name": "hasProfit", "type": "bool"},
            {"internalType": "uint256", "name": "delta", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]
gmx_vault_contract = w3.eth.contract(address=GMX_VAULT_ADDRESS, abi=GMX_VAULT_ABI)

# --- Pydantic Models ---
class AgentRequest(BaseModel):
    agent_name: str
    prompt: str
    soulbound_id: str | None = None

class UpgradeSuggestionRequest(BaseModel):
    agent_name: str
    suggestion_type: str
    suggestion: str

# --- Core State ---
@dataclass
class State:
    pulses: Dict[str, "Pulse"] = field(default_factory=dict)
    agents: Dict[str, GenesisAgent] = field(default_factory=dict)
    identities: Dict[str, SoulBoundIdentity] = field(default_factory=dict)

state = State()

# --- Pulse System ---
@dataclass
class Action:
    type: str
    data: Dict[str, Any]

@dataclass
class Pulse:
    name: str
    interval: float
    next_tick: float
    actions: List[Action]
    logger: TCCLogger

    def fire(self):
        for action in self.actions:
            self.logger.log("pulse", self.name.encode(), json.dumps(action.data).encode())
            if self.name == "reflection_pulse":
                for agent in state.agents.values():
                    agent.reflect_on_reflections()
            elif self.name == "gmx_liquidation_scanner":
                run_liquidation_scan()

# --- Liquidation Scanner ---
def run_liquidation_scan():
    accounts = [
        "0x84a08C0F2161aefb905bCb8B3eBa3c9C89bE2392",
        "0x1268b6bF663a1C2c2B8D6d00A14aBC83Bd821C90"
    ]
    token = Web3.to_checksum_address("0x82af49447d8a07e3bd95bd0d56f35241523fbab1")
    errors = []
    for account in accounts:
        try:
            pos = gmx_vault_contract.functions.getPosition(account, token, token, True).call()
            size, collateral, _, _, _, _, has_profit, delta = pos
            if size > 0 and not has_profit and delta > 0.9 * collateral:
                logger.log("liquidation_alert", account.encode(), "🔥 Risky".encode("utf-8"), {"collateral": collateral})
        except Exception as e:
            errors.append(f"{account}: {str(e)}")
    if errors:
        combined_error = " | ".join(errors)
        logger.log("scan_error", b"multiple_accounts", combined_error.encode(), {}, "ERROR", "GMX_SCAN_FAIL")

# --- Main Loop ---
def start_loop():
    while True:
        now = time.time()
        for pulse in state.pulses.values():
            if now >= pulse.next_tick:
                pulse.fire()
                pulse.next_tick = now + pulse.interval
        time.sleep(0.2)

# --- FastAPI Routes ---
@app.get("/ping")
async def ping():
    return {"status": "alive"}

@app.post("/agent")
async def agent_endpoint(req: AgentRequest):
    if req.soulbound_id and req.soulbound_id not in state.identities:
        raise HTTPException(status_code=403, detail="Invalid SoulBound ID")
    agent = state.agents.get(req.agent_name)
    if not agent:
        agent = GenesisAgent(name=req.agent_name, logger=TCCLogger())
        state.agents[req.agent_name] = agent
    response = agent.think(req.prompt)
    return {"response": response, "agent": req.agent_name}

@app.post("/suggest")
async def log_upgrade_suggestion(req: UpgradeSuggestionRequest):
    suggestion = UpgradeSuggestion(
        agent_name=req.agent_name,
        suggestion_type=req.suggestion_type,
        suggestion=req.suggestion
    )
    state.upgrade_suggestions.append(suggestion)
    logger.log("upgrade_suggestion", req.agent_name.encode(), req.suggestion.encode(), {"type": req.suggestion_type})
    return {"status": "suggestion logged", "timestamp": suggestion.timestamp}

@app.get("/logs")
async def get_logs():
    logs = [{
        "step": entry.step,
        "operation": entry.operation,
        "log_level": entry.log_level,
        "error_code": entry.error_code,
        "metadata": entry.metadata,
        "timestamp": entry.timestamp,
    } for entry in state.logger.logs]
    return {"logs": logs}

# --- Initialization ---
def setup_pulses():
    state.pulses["reflection_pulse"] = Pulse(
        name="reflection_pulse",
        interval=10,
        next_tick=time.time() + 10,
        actions=[Action(type="reflect", data={})],
        logger=state.logger
    )
    state.pulses["gmx_liquidation_scanner"] = Pulse(
        name="gmx_liquidation_scanner",
        interval=15,
        next_tick=time.time() + 15,
        actions=[Action(type="scan", data={})],
        logger=state.logger
    )

if __name__ == "__main__":
    import uvicorn
    state.logger = TCCLogger()
    setup_pulses()
    threading.Thread(target=start_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)