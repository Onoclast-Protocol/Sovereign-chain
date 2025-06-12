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

import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
from .security import BioInspiredSecuritySystem
from .soulbound_identity import SoulBoundIdentitySystem, SoulBoundIdentity
from ..agents.tcc_logger import TCCLogger

# Configure logging
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("onoclast_yield_protocol")

# Environment variables
ARBITRUM_RPC_URL = "https://arb1.arbitrum.io/rpc"
YIELD_CONTRACT_ADDRESS = Web3.to_checksum_address("0xYourYieldContractAddress")  # Placeholder
ADMIN_PRIVATE_KEY = "0xYourAdminPrivateKey"  # Placeholder, store securely

# Yield Protocol ABI
YIELD_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "accrueKarma",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"},
        ],
        "name": "applyDecay",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
        "name": "getKarma",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

@dataclass
class KarmaTransaction:
    account: str
    amount: int
    action_type: str  # e.g., "post", "comment", "upvote", "validation"
    timestamp: int
    signature: str

@dataclass
class KarmaState:
    last_updated: int
    karma: int
    decay_rate: float  # Daily decay rate (e.g., 0.01 for 1% daily decay)

class YieldProtocol:
    KARMA_PER_POST = 50
    KARMA_PER_COMMENT = 20
    KARMA_PER_UPVOTE = 10
    KARMA_PER_VALIDATION = 30
    KARMA_PER_VOUCHER_CREATION = 50
    KARMA_PER_VOUCHER_REDEMPTION = 20
    KARMA_PER_PULSE_CREATION = 200
    KARMA_PER_SIGNAL = 50
    DEFAULT_DECAY_RATE = 0.01  # 1% daily decay
    WEEKLY_BASIC_KARMA = 100  # Weekly karma for inclusion

    def __init__(self, soulbound_system: SoulBoundIdentitySystem, ipfs_endpoint: str = "/ip4/127.0.0.1/tcp/5001", log_level: str = "INFO"):
        """Initialize Yield Protocol with SoulBound Identity, IPFS, Web3, security, and logging."""
        self.logger = TCCLogger(level=log_level)
        self.security = BioInspiredSecuritySystem()
        self.security.establish_baseline()
        self.soulbound_system = soulbound_system
        try:
            import ipfshttpclient
            self.ipfs_client = ipfshttpclient.connect(ipfs_endpoint)
        except Exception as e:
            self.logger.log(
                "ipfs_init",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "IPFS_INIT_FAILED"
            )
            raise RuntimeError(f"Failed to connect to IPFS: {e}")

        self.w3 = Web3(Web3.HTTPProvider(ARBITRUM_RPC_URL, request_kwargs={"timeout": 30}))
        if not self.w3.is_connected():
            self.logger.log(
                "web3_init",
                b"",
                b"",
                {"error": "Web3 connection failed"},
                "ERROR",
                "WEB3_INIT_FAILED"
            )
            raise RuntimeError("Web3 connection failed")

        self.contract = self.w3.eth.contract(address=YIELD_CONTRACT_ADDRESS, abi=YIELD_ABI)
        self.admin_account = Account.from_key(ADMIN_PRIVATE_KEY)
        self.karma_states: Dict[str, KarmaState] = {}  # account -> KarmaState
        self.karma_transactions: List[KarmaTransaction] = []
        self._load_karma_states()

    def _load_karma_states(self):
        """Load karma states from data/karma_states.json."""
        try:
            with open("data/karma_states.json", "r", encoding="utf-8") as f:
                states_data = json.load(f)
            for state_data in states_data:
                self.karma_states[state_data["account"]] = KarmaState(
                    last_updated=state_data["last_updated"],
                    karma=state_data["karma"],
                    decay_rate=state_data["decay_rate"]
                )
        except FileNotFoundError:
            self.logger.log("load_karma_states", b"", b"", {"info": "No karma states file found, starting fresh"})
        except Exception as e:
            self.logger.log(
                "load_karma_states",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "LOAD_KARMA_STATES_FAILED"
            )

    def _save_karma_states(self):
        """Save karma states to data/karma_states.json."""
        try:
            with open("data/karma_states.json", "w", encoding="utf-8") as f:
                json.dump([
                    {
                        "account": account,
                        "last_updated": state.last_updated,
                        "karma": state.karma,
                        "decay_rate": state.decay_rate
                    }
                    for account, state in self.karma_states.items()
                ], f, indent=2)
        except Exception as e:
            self.logger.log(
                "save_karma_states",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "SAVE_KARMA_STATES_FAILED"
            )
            raise

    def _store_on_ipfs(self, content: str) -> str:
        """Store content on IPFS and return CID."""
        start_time = time.time_ns()
        try:
            result = self.ipfs_client.add_bytes(content.encode("utf-8"))
            cid = result["Hash"]
            self.logger.log(
                "ipfs_store",
                content.encode(),
                cid.encode(),
                {"execution_time_ns": time.time_ns() - start_time}
            )
            return cid
        except Exception as e:
            self.logger.log(
                "ipfs_store",
                content.encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "IPFS_STORE_FAILED"
            )
            raise

    def _sign_transaction(self, tx: Dict) -> Dict:
        """Sign a Web3 transaction with admin account."""
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

    def _apply_decay(self, account: str, current_time: int) -> int:
        """Apply karma decay based on elapsed time."""
        if account not in self.karma_states:
            return 0
        state = self.karma_states[account]
        days_elapsed = (current_time - state.last_updated) / (24 * 3600)
        if days_elapsed <= 0:
            return state.karma
        decayed_karma = int(state.karma * (1 - state.decay_rate) ** days_elapsed)
        state.karma = max(0, decayed_karma)
        state.last_updated = current_time

        # Build transaction
        nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
        tx = self.contract.functions.applyDecay(account).build_transaction({
            "from": self.admin_account.address,
            "nonce": nonce,
            "gas": 100000,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": 42161,  # Arbitrum One
        })
        self._sign_transaction(tx)

        self._save_karma_states()
        self.logger.log(
            "apply_decay",
            account.encode(),
            str(state.karma).encode(),
            {"days_elapsed": days_elapsed, "new_karma": state.karma}
        )
        return state.karma

    def accrue_karma(self, account: str, amount: int, action_type: str, signature: str) -> int:
        """Accrue karma for an account based on an action."""
        start_time = time.time_ns()
        try:
            # Verify identity
            identity_data = self.soulbound_system.get_identity(account)
            if not identity_data:
                raise ValueError("Invalid SoulBound Identity")

            # Verify signature
            message = encode_defunct(text=f"{action_type}:{amount}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != account.lower():
                raise ValueError("Invalid signature")

            # Security check
            transaction_data = f"{account}:{action_type}:{amount}"
            is_anomaly, score, explanation = self.security.detect_anomaly(transaction_data.encode())
            if is_anomaly:
                self.logger.log(
                    "accrue_karma_anomaly",
                    transaction_data.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Initialize or update karma state
            current_time = int(time.time())
            if account not in self.karma_states:
                self.karma_states[account] = KarmaState(
                    last_updated=current_time,
                    karma=0,
                    decay_rate=self.DEFAULT_DECAY_RATE
                )
            self._apply_decay(account, current_time)

            # Accrue karma
            self.karma_states[account].karma += amount

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.accrueKarma(account, amount).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 150000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            self._sign_transaction(tx)

            # Log transaction
            transaction = KarmaTransaction(
                account=account,
                amount=amount,
                action_type=action_type,
                timestamp=current_time,
                signature=signature
            )
            self.karma_transactions.append(transaction)
            transaction_data = json.dumps({
                "account": account,
                "amount": amount,
                "action_type": action_type,
                "timestamp": current_time
            })
            cid = self._store_on_ipfs(transaction_data)

            # Update soulbound identity reputation
            identity: SoulBoundIdentity = self.soulbound_system.identities.get(account.lower())
            if identity:
                identity.metadata["karma"] = self.karma_states[account].karma
                self.soulbound_system._save_identities()

            self._save_karma_states()
            self.logger.log(
                "accrue_karma",
                transaction_data.encode(),
                cid.encode(),
                {
                    "account": account,
                    "action_type": action_type,
                    "amount": amount,
                    "new_karma": self.karma_states[account].karma,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return self.karma_states[account].karma
        except Exception as e:
            self.logger.log(
                "accrue_karma",
                transaction_data.encode() if 'transaction_data' in locals() else b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "ACCRUE_KARMA_FAILED"
            )
            raise

    def distribute_weekly_karma(self, accounts: List[str]) -> Dict[str, int]:
        """Distribute weekly basic karma to all accounts."""
        start_time = time.time_ns()
        try:
            results = {}
            current_time = int(time.time())
            for account in accounts:
                identity_data = self.soulbound_system.get_identity(account)
                if not identity_data:
                    continue
                signature = self.w3.eth.account.sign_message(
                    encode_defunct(text=f"weekly_karma:{current_time}"),
                    private_key=self.admin_account.key
                ).signature.hex()
                karma = self.accrue_karma(
                    account=account,
                    amount=self.WEEKLY_BASIC_KARMA,
                    action_type="weekly_distribution",
                    signature=signature
                )
                results[account] = karma
            self.logger.log(
                "distribute_weekly_karma",
                json.dumps(accounts).encode(),
                json.dumps(results).encode(),
                {
                    "account_count": len(accounts),
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return results
        except Exception as e:
            self.logger.log(
                "distribute_weekly_karma",
                json.dumps(accounts).encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "WEEKLY_KARMA_FAILED"
            )
            raise

    def get_karma(self, account: str) -> int:
        """Retrieve current karma for an account, applying decay."""
        start_time = time.time_ns()
        try:
            identity_data = self.soulbound_system.get_identity(account)
            if not identity_data:
                raise ValueError("Invalid SoulBound Identity")
            current_time = int(time.time())
            karma = self._apply_decay(account, current_time)
            self.logger.log(
                "get_karma",
                account.encode(),
                str(karma).encode(),
                {"account": account, "karma": karma, "execution_time_ns": time.time_ns() - start_time}
            )
            return karma
        except Exception as e:
            self.logger.log(
                "get_karma",
                account.encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "GET_KARMA_FAILED"
            )
            raise