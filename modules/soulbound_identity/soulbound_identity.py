"""
Dual License

For Open-Source Individuals:
MIT License

Copyright (c) 2024 James B. Chapman

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
import hashlib
import base64
import logging
import os
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import ipfshttpclient
from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
from datetime import datetime
from .security import BioInspiredSecuritySystem
from ..agents.tcc_logger import TCCLogger, TCCKeccakEngine, EntropyCoordinator

# Configure logging
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("onoclast_soulbound_identity")

# Environment variables
ARBITRUM_RPC_URL = "https://arb1.arbitrum.io/rpc"
SOULBOUND_CONTRACT_ADDRESS = Web3.to_checksum_address("0xYourSoulBoundContractAddress")  # Placeholder
ADMIN_PRIVATE_KEY = "0xYourAdminPrivateKey"  # Placeholder, store securely
IDENTITIES_FILE = "data/identities.json"
IPFS_ENDPOINT = "/ip4/127.0.0.1/tcp/5001"
ENTROPY_FEE = 1000  # Default fee per entropy reveal

# SoulBound Identity ABI
SOULBOUND_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "string", "name": "metadataCid", "type": "string"},
        ],
        "name": "mintIdentity",
        "outputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "tokenId", "type": "uint256"},
            {"internalType": "string", "name": "metadataCid", "type": "string"},
        ],
        "name": "updateMetadata",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "owner", "type": "address"}],
        "name": "getTokenIdByOwner",
        "outputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "name": "getIdentity",
        "outputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "string", "name": "metadataCid", "type": "string"},
            {"internalType": "uint256", "name": "createdAt", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

@dataclass
class SoulBoundIdentity:
    token_id: int
    owner: str
    metadata_cid: str
    created_at: int
    metadata: Dict = field(default_factory=dict)

class SoulBoundIdentitySystem:
    def __init__(self, ipfs_endpoint: str = IPFS_ENDPOINT, log_level: str = "INFO"):
        """Initialize SoulBound Identity system with IPFS, Web3, security, logging, and entropy."""
        self.logger = TCCLogger()
        self.security = BioInspiredSecuritySystem()
        self.security.establish_baseline()
        try:
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
        
        self.contract = self.w3.eth.contract(address=SOULBOUND_CONTRACT_ADDRESS, abi=SOULBOUND_ABI)
        self.admin_account = Account.from_key(ADMIN_PRIVATE_KEY)
        self.identities: Dict[str, SoulBoundIdentity] = {}  # owner_address -> identity
        
        # Initialize entropy engines
        self.engine_a = TCCKeccakEngine(fee_per_entropy=ENTROPY_FEE)
        self.engine_b = TCCKeccakEngine(fee_per_entropy=ENTROPY_FEE)
        self.engine_c = TCCKeccakEngine(fee_per_entropy=ENTROPY_FEE)
        self.entropy_coordinator = EntropyCoordinator(self.engine_a, self.engine_b, self.engine_c)
        
        self._load_identities()

    def _validate_birthdate(self, birthdate: str) -> bool:
        """Validate birthdate format and reasonableness (YYYY-MM-DD)."""
        try:
            date = datetime.strptime(birthdate, "%Y-%m-%d")
            current_year = datetime.now().year
            if not (1900 <= date.year <= current_year):
                return False
            return True
        except ValueError:
            return False

    def _compute_dna_strain(self, metadata: Dict) -> str:
        """Compute DNA strain as SHA-256 hash of all metadata."""
        metadata_str = json.dumps(metadata, sort_keys=True)
        return hashlib.sha256(metadata_str.encode('utf-8')).hexdigest()

    def _load_identities(self):
        """Load identities from data/identities.json."""
        try:
            if not os.path.exists(IDENTITIES_FILE):
                self.logger.log(
                    "load_identities",
                    b"",
                    b"",
                    {"info": "No identities file found, starting fresh"},
                    "INFO",
                    "SUCCESS"
                )
                return
            with open(IDENTITIES_FILE, "r", encoding="utf-8") as f:
                identities_data = json.load(f)
            for identity_data in identities_data:
                identity = SoulBoundIdentity(**identity_data)
                self.identities[identity.owner.lower()] = identity
        except Exception as e:
            self.logger.log(
                "load_identities",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "LOAD_IDENTITIES_FAILED"
            )

    def _save_identities(self):
        """Save identities to data/identities.json."""
        os.makedirs(os.path.dirname(IDENTITIES_FILE), exist_ok=True)
        try:
            with open(IDENTITIES_FILE, "w", encoding="utf-8") as f:
                json.dump([vars(identity) for identity in self.identities.values()], f, indent=2)
            self.logger.log(
                "save_identities",
                b"",
                b"",
                {"identity_count": len(self.identities)},
                "INFO",
                "SUCCESS"
            )
        except Exception as e:
            self.logger.log(
                "save_identities",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "SAVE_IDENTITIES_FAILED"
            )
            raise

    def _store_on_ipfs(self, content: bytes, content_type: str = "json") -> str:
        """Store content on IPFS and return CID."""
        start_time = time.time_ns()
        try:
            result = self.ipfs_client.add_bytes(content)
            cid = result["Hash"]
            self.logger.log(
                "ipfs_store",
                base64.b64encode(content)[:100],
                cid.encode(),
                {"content_type": content_type, "execution_time_ns": time.time_ns() - start_time},
                "INFO",
                "SUCCESS"
            )
            return cid
        except Exception as e:
            self.logger.log(
                "ipfs_store",
                base64.b64encode(content)[:100],
                b"",
                {"error": str(e), "content_type": content_type, "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "IPFS_STORE_FAILED"
            )
            raise

    def _retrieve_from_ipfs(self, cid: str) -> bytes:
        """Retrieve content from IPFS by CID."""
        start_time = time.time_ns()
        try:
            content = self.ipfs_client.cat(cid)
            self.logger.log(
                "ipfs_retrieve",
                cid.encode(),
                base64.b64encode(content)[:100],
                {"execution_time_ns": time.time_ns() - start_time},
                "INFO",
                "SUCCESS"
            )
            return content
        except Exception as e:
            self.logger.log(
                "ipfs_retrieve",
                cid.encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "IPFS_RETRIEVE_FAILED"
            )
            raise

    def _sign_transaction(self, tx: Dict) -> Dict:
        """Sign a Web3 transaction with admin account."""
        start_time = time.time_ns()
        try:
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.admin_account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if receipt.status != 1:
                raise RuntimeError("Transaction failed")
            self.logger.log(
                "sign_transaction",
                json.dumps(tx).encode(),
                tx_hash.hex().encode(),
                {"execution_time_ns": time.time_ns() - start_time},
                "INFO",
                "SUCCESS"
            )
            return receipt
        except Exception as e:
            self.logger.log(
                "sign_transaction",
                json.dumps(tx).encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "SIGN_TX_FAILED"
            )
            raise

    def mint_identity(self, owner_address: str, metadata: Dict, signature: str, entropy_commitment: bytes) -> int:
        """Mint a new SoulBound Identity with entropy commitment."""
        start_time = time.time_ns()
        try:
            # Validate metadata
            if not all(key in metadata for key in ["birthdate", "public_key", "zk_did"]):
                raise ValueError("Metadata missing required fields: birthdate, public_key, zk_did")
            if not self._validate_birthdate(metadata["birthdate"]):
                raise ValueError("Invalid birthdate format or range")
            if owner_address.lower() in self.identities:
                raise ValueError("Identity already exists for this address")

            # Verify signature
            metadata_str = json.dumps(metadata, sort_keys=True)
            message = encode_defunct(text=metadata_str)
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != owner_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            is_anomaly, score, explanation = self.security.detect_anomaly(metadata_str.encode())
            if is_anomaly:
                self.logger.log(
                    "mint_identity_anomaly",
                    metadata_str.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Commit entropy
            self.entropy_coordinator.commit_entropy_all(owner_address, entropy_commitment, entropy_commitment, entropy_commitment)
            metadata["entropy_commitments"] = [entropy_commitment.hex()]
            metadata["social_media"] = metadata.get("social_media", [])
            metadata["created_at"] = int(time.time())
            metadata["dna_strain"] = self._compute_dna_strain(metadata)

            # Store metadata on IPFS
            cid = self._store_on_ipfs(json.dumps(metadata).encode('utf-8'))

            # Mint SBT on Arbitrum
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.mintIdentity(owner_address, cid).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 250000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            receipt = self._sign_transaction(tx)
            token_id = receipt.logs[0]['topics'][3].hex()  # Simplified, assumes IdentityMinted event

            # Create and store identity
            identity = SoulBoundIdentity(
                token_id=token_id,
                owner=owner_address,
                metadata_cid=cid,
                created_at=int(time.time()),
                metadata=metadata,
            )
            self.identities[owner_address.lower()] = identity
            self._save_identities()

            self.logger.log(
                "mint_identity",
                metadata_str.encode(),
                json.dumps({"token_id": token_id, "cid": cid}).encode(),
                {"owner": owner_address, "execution_time_ns": time.time_ns() - start_time},
                "INFO",
                "SUCCESS"
            )
            return token_id
        except Exception as e:
            self.logger.log(
                "mint_identity",
                metadata_str.encode() if 'metadata_str' in locals() else b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "MINT_IDENTITY_FAILED"
            )
            raise

    def update_metadata(self, owner_address: str, new_metadata: Dict, signature: str, entropy_commitment: bytes) -> None:
        """Update metadata with new entropy commitment."""
        start_time = time.time_ns()
        try:
            if owner_address.lower() not in self.identities:
                raise ValueError("Identity not found")
            identity = self.identities[owner_address.lower()]

            # Validate new metadata
            if not all(key in new_metadata for key in ["birthdate", "public_key", "zk_did"]):
                raise ValueError("Metadata missing required fields: birthdate, public_key, zk_did")
            if not self._validate_birthdate(new_metadata["birthdate"]):
                raise ValueError("Invalid birthdate format or range")

            # Verify signature and multi-factor ownership
            metadata_str = json.dumps(new_metadata, sort_keys=True)
            message = encode_defunct(text=metadata_str)
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != owner_address.lower():
                raise ValueError("Invalid signature")
            if new_metadata["birthdate"] != identity.metadata["birthdate"]:
                raise ValueError("Birthdate mismatch")
            if new_metadata["zk_did"] != identity.metadata["zk_did"]:
                raise ValueError("zk-DID mismatch")

            # Security check
            is_anomaly, score, explanation = self.security.detect_anomaly(metadata_str.encode())
            if is_anomaly:
                self.logger.log(
                    "update_metadata_anomaly",
                    metadata_str.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Commit new entropy
            self.entropy_coordinator.commit_entropy_all(owner_address, entropy_commitment, entropy_commitment, entropy_commitment)
            new_metadata["entropy_commitments"] = identity.metadata.get("entropy_commitments", []) + [entropy_commitment.hex()]
            new_metadata["social_media"] = new_metadata.get("social_media", identity.metadata["social_media"])
            new_metadata["updated_at"] = int(time.time())
            new_metadata["dna_strain"] = self._compute_dna_strain(new_metadata)

            # Store new metadata on IPFS
            cid = self._store_on_ipfs(json.dumps(new_metadata).encode('utf-8'))

            # Update SBT on Arbitrum
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.updateMetadata(identity.token_id, cid).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 150000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            self._sign_transaction(tx)

            # Update identity
            identity.metadata_cid = cid
            identity.metadata = new_metadata
            self._save_identities()

            self.logger.log(
                "update_metadata",
                metadata_str.encode(),
                cid.encode(),
                {"owner": owner_address, "token_id": identity.token_id, "execution_time_ns": time.time_ns() - start_time},
                "INFO",
                "SUCCESS"
            )
        except Exception as e:
            self.logger.log(
                "update_metadata",
                metadata_str.encode() if 'metadata_str' in locals() else b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "UPDATE_METADATA_FAILED"
            )
            raise

    def upload_social_media(self, owner_address: str, content: bytes, content_type: str, signature: str, entropy_commitment: bytes) -> str:
        """Upload social media content to IPFS with entropy commitment."""
        start_time = time.time_ns()
        try:
            if owner_address.lower() not in self.identities:
                raise ValueError("Identity not found")
            identity = self.identities[owner_address.lower()]

            # Verify signature
            content_hash = hashlib.sha256(content).hexdigest()
            message = encode_defunct(text=f"upload_social_media:{content_hash}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != owner_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            is_anomaly, score, explanation = self.security.detect_anomaly(content)
            if is_anomaly:
                self.logger.log(
                    "upload_social_media_anomaly",
                    content_hash.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Commit entropy
            self.entropy_coordinator.commit_entropy_all(owner_address, entropy_commitment, entropy_commitment, entropy_commitment)

            # Store content on IPFS
            cid = self._store_on_ipfs(content, content_type)

            # Update social media metadata
            social_media_entry = {
                "cid": cid,
                "type": content_type,
                "timestamp": int(time.time()),
                "hash": content_hash
            }
            identity.metadata["social_media"].append(social_media_entry)
            identity.metadata["entropy_commitments"] = identity.metadata.get("entropy_commitments", []) + [entropy_commitment.hex()]
            identity.metadata["dna_strain"] = self._compute_dna_strain(identity.metadata)

            # Store updated metadata on IPFS
            new_metadata_cid = self._store_on_ipfs(json.dumps(identity.metadata).encode('utf-8'))

            # Update SBT on Arbitrum
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.updateMetadata(identity.token_id, new_metadata_cid).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 150000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            self._sign_transaction(tx)

            # Update identity
            identity.metadata_cid = new_metadata_cid
            self._save_identities()

            self.logger.log(
                "upload_social_media",
                content_hash.encode(),
                cid.encode(),
                {"owner": owner_address, "content_type": content_type, "execution_time_ns": time.time_ns() - start_time},
                "INFO",
                "SUCCESS"
            )
            return cid
        except Exception as e:
            self.logger.log(
                "upload_social_media",
                content_hash.encode() if 'content_hash' in locals() else b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "UPLOAD_SOCIAL_MEDIA_FAILED"
            )
            raise

    def verify_identity(self, owner_address: str, signature: str, zk_did: str, birthdate: str, entropy: bytes, fee: int) -> bool:
        """Verify identity with multi-factor ownership and entropy reveal."""
        start_time = time.time_ns()
        try:
            if owner_address.lower() not in self.identities:
                self.logger.log(
                    "verify_identity",
                    owner_address.encode(),
                    b"false",
                    {"owner": owner_address, "reason": "Identity not found"},
                    "INFO",
                    "NO_IDENTITY"
                )
                return False
            identity = self.identities[owner_address.lower()]

            # Verify signature
            message = encode_defunct(text=f"verify_identity:{owner_address}:{zk_did}:{birthdate}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != owner_address.lower():
                self.logger.log(
                    "verify_identity",
                    owner_address.encode(),
                    b"false",
                    {"owner": owner_address, "reason": "Invalid signature"},
                    "INFO",
                    "INVALID_SIGNATURE"
                )
                return False

            # Verify zk-DID and birthdate
            if zk_did != identity.metadata["zk_did"] or birthdate != identity.metadata["birthdate"]:
                self.logger.log(
                    "verify_identity",
                    owner_address.encode(),
                    b"false",
                    {"owner": owner_address, "reason": "zk-DID or birthdate mismatch"},
                    "INFO",
                    "MISMATCH"
                )
                return False

            # Verify DNA strain
            metadata_copy = identity.metadata.copy()
            metadata_copy["dna_strain"] = self._compute_dna_strain(metadata_copy)
            if metadata_copy["dna_strain"] != identity.metadata["dna_strain"]:
                self.logger.log(
                    "verify_identity",
                    owner_address.encode(),
                    b"false",
                    {"owner": owner_address, "reason": "DNA strain mismatch"},
                    "INFO",
                    "DNA_STRAIN_MISMATCH"
                )
                return False

            # Verify entropy
            self.entropy_coordinator.reveal_entropy_all(owner_address, entropy, entropy, entropy, fee)
            combined_entropy = self.entropy_coordinator.get_combined_entropy()
            if hashlib.sha256(entropy).hexdigest() not in identity.metadata.get("entropy_commitments", []):
                self.logger.log(
                    "verify_identity",
                    owner_address.encode(),
                    b"false",
                    {"owner": owner_address, "reason": "Invalid entropy commitment"},
                    "INFO",
                    "INVALID_ENTROPY"
                )
                return False

            self.logger.log(
                "verify_identity",
                owner_address.encode(),
                b"true",
                {"owner": owner_address, "combined_entropy": combined_entropy.hex(), "execution_time_ns": time.time_ns() - start_time},
                "INFO",
                "SUCCESS"
            )
            return True
        except Exception as e:
            self.logger.log(
                "verify_identity",
                owner_address.encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "VERIFY_IDENTITY_FAILED"
            )
            raise

    def get_identity(self, owner_address: str) -> Dict:
        """Retrieve identity details for an owner."""
        start_time = time.time_ns()
        try:
            if owner_address.lower() not in self.identities:
                raise ValueError("Identity not found")
            identity = self.identities[owner_address.lower()]
            identity_data = {
                "token_id": identity.token_id,
                "owner": identity.owner,
                "metadata_cid": identity.metadata_cid,
                "created_at": identity.created_at,
                "metadata": identity.metadata,
            }
            self.logger.log(
                "get_identity",
                owner_address.encode(),
                json.dumps(identity_data).encode(),
                {"owner": owner_address, "execution_time_ns": time.time_ns() - start_time},
                "INFO",
                "SUCCESS"
            )
            return identity_data
        except Exception as e:
            self.logger.log(
                "get_identity",
                owner_address.encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "GET_IDENTITY_FAILED"
            )
            raise

    def get_social_media(self, owner_address: str) -> List[Dict]:
        """Retrieve social media content for an owner."""
        start_time = time.time_ns()
        try:
            if owner_address.lower() not in self.identities:
                raise ValueError("Identity not found")
            social_media = self.identities[owner_address.lower()].metadata.get("social_media", [])
            self.logger.log(
                "get_social_media",
                owner_address.encode(),
                json.dumps(social_media).encode(),
                {"owner": owner_address, "content_count": len(social_media), "execution_time_ns": time.time_ns() - start_time},
                "INFO",
                "SUCCESS"
            )
            return social_media
        except Exception as e:
            self.logger.log(
                "get_social_media",
                owner_address.encode(),
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "GET_SOCIAL_MEDIA_FAILED"
            )
            raise

    def save_entropy_log(self, filename: str) -> None:
        """Save entropy coordinator log to a file."""
        self.entropy_coordinator.save_log(filename)