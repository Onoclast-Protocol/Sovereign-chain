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
from .soulbound_identity import SoulBoundIdentitySystem
from .yield_protocol import YieldProtocol
from .town_square import TownSquare
from .messaging import Messaging
from ..agents.tcc_logger import TCCLogger

# Configure logging
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("onoclast_consensus")

# Environment variables
ARBITRUM_RPC_URL = "https://arb1.arbitrum.io/rpc"
CONSENSUS_CONTRACT_ADDRESS = Web3.to_checksum_address("0xYourConsensusContractAddress")  # Placeholder
ADMIN_PRIVATE_KEY = "0xYourAdminPrivateKey"  # Placeholder, store securely

# Consensus Contract ABI (simplified)
CONSENSUS_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "claimId", "type": "uint256"},
            {"internalType": "address", "name": "validator", "type": "address"},
            {"internalType": "bool", "name": "isValid", "type": "bool"},
            {"internalType": "string", "name": "validationCid", "type": "string"},
        ],
        "name": "submitValidation",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "claimId", "type": "uint256"},
            {"internalType": "string", "name": "resultCid", "type": "string"},
        ],
        "name": "finalizeConsensus",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "claimId", "type": "uint256"}],
        "name": "getConsensusResult",
        "outputs": [
            {"internalType": "bool", "name": "isValid", "type": "bool"},
            {"internalType": "uint256", "name": "totalKarma", "type": "uint256"},
            {"internalType": "string", "name": "resultCid", "type": "string"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

@dataclass
class Validation:
    validator: str
    is_valid: bool
    comment_id: Optional[int]  # Comment ID in TownSquare, if applicable
    timestamp: int
    cid: str

@dataclass
class Claim:
    claim_id: int
    post_id: int
    group_id: int  # Messaging group for validator coordination
    validations: List[Validation] = field(default_factory=list)
    is_finalized: bool = False
    result: Optional[bool] = None
    total_karma: int = 0
    result_cid: str = ""
    timestamp: int = 0

class Consensus:
    KARMA_PER_VALIDATION = 30
    MIN_KARMA_TO_VALIDATE = 200
    CONSENSUS_THRESHOLD = 0.7  # 70% agreement required
    MIN_VALIDATORS = 3
    CONSENSUS_TIMEOUT = 7 * 24 * 3600  # 7 days

    def __init__(
        self,
        soulbound_system: SoulBoundIdentitySystem,
        yield_protocol: YieldProtocol,
        town_square: TownSquare,
        messaging: Messaging,
        ipfs_endpoint: str = "/ip4/127.0.0.1/tcp/5001",
        log_level: str = "INFO",
    ):
        """Initialize Consensus with dependencies."""
        self.logger = TCCLogger(level=log_level)
        self.security = BioInspiredSecuritySystem()
        self.security.establish_baseline()
        self.soulbound_system = soulbound_system
        self.yield_protocol = yield_protocol
        self.town_square = town_square
        self.messaging = messaging
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

        self.contract = self.w3.eth.contract(address=CONSENSUS_CONTRACT_ADDRESS, abi=CONSENSUS_ABI)
        self.admin_account = Account.from_key(ADMIN_PRIVATE_KEY)
        self.claims: Dict[int, Claim] = {}
        self._load_claims()

    def _load_claims(self):
        """Load claims from data/claims.json."""
        try:
            with open("data/claims.json", "r", encoding="utf-8") as f:
                claims_data = json.load(f)
            for claim_data in claims_data:
                self.claims[claim_data["claim_id"]] = Claim(**claim_data)
        except FileNotFoundError:
            self.logger.log("load_claims", b"", b"", {"info": "No claims file found, starting fresh"})
        except Exception as e:
            self.logger.log(
                "load_claims",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "LOAD_CLAIMS_FAILED"
            )

    def _save_claims(self):
        """Save claims to data/claims.json."""
        try:
            with open("data/claims.json", "w", encoding="utf-8") as f:
                json.dump([vars(claim) for claim in self.claims.values()], f, indent=2)
        except Exception as e:
            self.logger.log(
                "save_claims",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "SAVE_CLAIMS_FAILED"
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

    def start_claim_validation(self, post_id: int, admin_address: str, signature: str) -> int:
        """Start a validation process for a TownSquare post."""
        start_time = time.time_ns()
        try:
            # Verify post and admin
            if post_id not in self.town_square.posts:
                raise ValueError("Post not found")
            if not self.soulbound_system.verify_identity(admin_address):
                raise ValueError("Invalid admin identity")
            karma = self.yield_protocol.get_karma(admin_address)
            if karma < self.MIN_KARMA_TO_VALIDATE:
                raise ValueError(f"Admin karma {karma} below required {self.MIN_KARMA_TO_VALIDATE}")

            # Verify signature
            message = encode_defunct(text=f"start_validation:{post_id}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != admin_address.lower():
                raise ValueError("Invalid signature")

            # Create validator group via Messaging
            group_id = self.messaging.create_group(
                admin_address=admin_address,
                min_karma=self.MIN_KARMA_TO_VALIDATE,
                signature=signature
            )

            # Store claim metadata on IPFS
            claim_data = {
                "post_id": post_id,
                "admin": admin_address,
                "group_id": group_id,
                "timestamp": int(time.time()),
            }
            cid = self._store_on_ipfs(json.dumps(claim_data))

            # Build transaction (assuming claim_id is generated on-chain)
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.createClaim(cid, post_id).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,  # Arbitrum One
            })
            receipt = self._sign_transaction(tx)
            claim_id = self.contract.events.ClaimCreated().process_receipt(receipt)[0]["args"]["claimId"]

            # Create claim object
            claim = Claim(
                claim_id=claim_id,
                post_id=post_id,
                group_id=group_id,
                timestamp=int(time.time()),
                result_cid=cid,
            )
            self.claims[claim_id] = claim
            self._save_claims()

            # Log action
            self.logger.log(
                "start_claim_validation",
                json.dumps(claim_data).encode(),
                cid.encode(),
                {
                    "claim_id": claim_id,
                    "post_id": post_id,
                    "group_id": group_id,
                    "admin": admin_address,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return claim_id
        except Exception as e:
            self.logger.log(
                "start_claim_validation",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "START_CLAIM_VALIDATION_FAILED"
            )
            raise

    def submit_validation(self, claim_id: int, validator_address: str, is_valid: bool, comment_id: Optional[int], signature: str) -> None:
        """Submit a validation for a claim."""
        start_time = time.time_ns()
        try:
            if claim_id not in self.claims:
                raise ValueError("Claim not found")
            claim = self.claims[claim_id]
            if claim.is_finalized:
                raise ValueError("Claim already finalized")
            if not self.soulbound_system.verify_identity(validator_address):
                raise ValueError("Invalid validator identity")
            karma = self.yield_protocol.get_karma(validator_address)
            if karma < self.MIN_KARMA_TO_VALIDATE:
                raise ValueError(f"Validator karma {karma} below required {self.MIN_KARMA_TO_VALIDATE}")
            if validator_address not in self.messaging.groups[claim.group_id].members:
                raise ValueError("Validator not in group")

            # Verify signature
            message = encode_defunct(text=f"validate:{claim_id}:{is_valid}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != validator_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            validation_data = f"{claim_id}:{validator_address}:{is_valid}"
            is_anomaly, score, explanation = self.security.detect_anomaly(validation_data.encode())
            if is_anomaly:
                self.logger.log(
                    "submit_validation_anomaly",
                    validation_data.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Store validation on IPFS
            validation_data = {
                "claim_id": claim_id,
                "validator": validator_address,
                "is_valid": is_valid,
                "comment_id": comment_id,
                "timestamp": int(time.time()),
            }
            cid = self._store_on_ipfs(json.dumps(validation_data))

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.submitValidation(claim_id, validator_address, is_valid, cid).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 150000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            self._sign_transaction(tx)

            # Add validation
            validation = Validation(
                validator=validator_address,
                is_valid=is_valid,
                comment_id=comment_id,
                timestamp=int(time.time()),
                cid=cid,
            )
            claim.validations.append(validation)
            self._save_claims()

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=validator_address,
                amount=self.KARMA_PER_VALIDATION,
                action_type="validation",
                signature=signature
            )

            # Log action
            self.logger.log(
                "submit_validation",
                json.dumps(validation_data).encode(),
                cid.encode(),
                {
                    "claim_id": claim_id,
                    "validator": validator_address,
                    "is_valid": is_valid,
                    "karma_earned": self.KARMA_PER_VALIDATION,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
        except Exception as e:
            self.logger.log(
                "submit_validation",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "SUBMIT_VALIDATION_FAILED"
            )
            raise

    def finalize_consensus(self, claim_id: int, admin_address: str, signature: str) -> bool:
        """Finalize consensus for a claim and distribute rewards."""
        start_time = time.time_ns()
        try:
            if claim_id not in self.claims:
                raise ValueError("Claim not found")
            claim = self.claims[claim_id]
            if claim.is_finalized:
                raise ValueError("Claim already finalized")
            if not self.soulbound_system.verify_identity(admin_address):
                raise ValueError("Invalid admin identity")
            if len(claim.validations) < self.MIN_VALIDATORS:
                raise ValueError(f"Insufficient validators: {len(claim.validations)} < {self.MIN_VALIDATORS}")
            if time.time() < claim.timestamp + self.CONSENSUS_TIMEOUT:
                raise ValueError("Consensus period not expired")

            # Verify signature
            message = encode_defunct(text=f"finalize:{claim_id}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != admin_address.lower():
                raise ValueError("Invalid signature")

            # Calculate consensus
            total_karma = 0
            valid_count = 0
            validator_karma = {}
            for validation in claim.validations:
                karma = self.yield_protocol.get_karma(validation.validator)
                total_karma += karma
                validator_karma[validation.validator] = karma
                if validation.is_valid:
                    valid_count += karma

            consensus_ratio = valid_count / total_karma if total_karma > 0 else 0
            is_valid = consensus_ratio >= self.CONSENSUS_THRESHOLD
            claim.is_finalized = True
            claim.result = is_valid
            claim.total_karma = total_karma

            # Distribute additional karma to aligned validators
            for validation in claim.validations:
                if validation.is_valid == is_valid:
                    self.yield_protocol.accrue_karma(
                        account=validation.validator,
                        amount=int(self.KARMA_PER_VALIDATION * (validator_karma[validation.validator] / total_karma)),
                        action_type="consensus_reward",
                        signature=signature
                    )

            # Store result on IPFS
            result_data = {
                "claim_id": claim_id,
                "post_id": claim.post_id,
                "is_valid": is_valid,
                "total_karma": total_karma,
                "consensus_ratio": consensus_ratio,
                "timestamp": int(time.time()),
            }
            claim.result_cid = self._store_on_ipfs(json.dumps(result_data))

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.finalizeConsensus(claim_id, claim.result_cid).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            self._sign_transaction(tx)

            # Update claim
            self._save_claims()

            # Log action
            self.logger.log(
                "finalize_consensus",
                json.dumps(result_data).encode(),
                claim.result_cid.encode(),
                {
                    "claim_id": claim_id,
                    "is_valid": is_valid,
                    "total_karma": total_karma,
                    "consensus_ratio": consensus_ratio,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return is_valid
        except Exception as e:
            self.logger.log(
                "finalize_consensus",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "FINALIZE_CONSENSUS_FAILED"
            )
            raise

    def get_consensus_result(self, claim_id: int) -> Dict:
        """Retrieve the consensus result for a claim."""
        start_time = time.time_ns()
        try:
            if claim_id not in self.claims:
                raise ValueError("Claim not found")
            claim = self.claims[claim_id]
            result_data = {
                "claim_id": claim.claim_id,
                "post_id": claim.post_id,
                "group_id": claim.group_id,
                "is_finalized": claim.is_finalized,
                "result": claim.result,
                "total_karma": claim.total_karma,
                "result_cid": claim.result_cid,
                "validations": [
                    {
                        "validator": v.validator,
                        "is_valid": v.is_valid,
                        "comment_id": v.comment_id,
                        "timestamp": v.timestamp,
                        "cid": v.cid,
                    }
                    for v in claim.validations
                ],
            }
            self.logger.log(
                "get_consensus_result",
                f"claim_id:{claim_id}".encode(),
                json.dumps(result_data).encode(),
                {"execution_time_ns": time.time_ns() - start_time}
            )
            return result_data
        except Exception as e:
            self.logger.log(
                "get_consensus_result",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "GET_CONSENSUS_RESULT_FAILED"
            )
            raise