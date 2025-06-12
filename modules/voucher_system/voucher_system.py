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
from typing import Dict, Optional
from dataclasses import dataclass, field
from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
from .security import BioInspiredSecuritySystem
from .soulbound_identity import SoulBoundIdentitySystem
from .yield_protocol import YieldProtocol
from .town_square import TownSquare
from .messaging import Messaging
from .consensus import Consensus
from ..agents.tcc_logger import TCCLogger
import ipfshttpclient

# Configure logging
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("onoclast_voucher_system")

# Chain configurations (example, extend as needed)
CHAIN_CONFIGS = {
    "arbitrum": {
        "rpc_url": "https://arb1.arbitrum.io/rpc",
        "contract_address": Web3.to_checksum_address("0xYourVoucherContractAddressArbitrum"),  # Placeholder
        "chain_id": 42161,
    },
    "ethereum": {
        "rpc_url": "https://mainnet.infura.io/v3/YOUR_INFURA_KEY",  # Placeholder
        "contract_address": Web3.to_checksum_address("0xYourVoucherContractAddressEthereum"),  # Placeholder
        "chain_id": 1,
    },
    # Add more chains as needed
}

# Voucher Contract ABI (based on PhysicalTokenTransfer.sol)
VOUCHER_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "token", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "bytes32", "name": "passwordHash", "type": "bytes32"},
        ],
        "name": "depositERC20",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "passwordHash", "type": "bytes32"},
        ],
        "name": "depositETH",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "voucherId", "type": "uint256"},
            {"internalType": "string", "name": "oldPassword", "type": "string"},
            {"internalType": "bytes32", "name": "newPasswordHash", "type": "bytes32"},
        ],
        "name": "transferVoucher",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "voucherId", "type": "uint256"},
            {"internalType": "string", "name": "password", "type": "string"},
            {"internalType": "address", "name": "recipient", "type": "address"},
        ],
        "name": "redeemVoucher",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "voucherId", "type": "uint256"}],
        "name": "vouchers",
        "outputs": [
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "bytes32", "name": "secretHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "transferCount", "type": "uint256"},
            {"internalType": "bool", "name": "redeemed", "type": "bool"},
            {"internalType": "bool", "name": "isEth", "type": "bool"},
            {"internalType": "address", "name": "tokenAddress", "type": "address"},
            {"internalType": "uint256", "name": "createdAt", "type": "uint256"},
            {"internalType": "uint256", "name": "lastTransferAt", "type": "uint256"},
            {"internalType": "address", "name": "depositor", "type": "address"},
            {"internalType": "address", "name": "redeemer", "type": "address"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "voucherId", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "netAmount", "type": "uint256"},
            {"indexed": False, "internalType": "bool", "name": "isEth", "type": "bool"},
            {"indexed": False, "internalType": "address", "name": "tokenAddress", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "createdAt", "type": "uint256"},
            {"indexed": False, "internalType": "address", "name": "depositor", "type": "address"},
        ],
        "name": "VoucherCreated",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "voucherId", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "transferCount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "lastTransferAt", "type": "uint256"},
        ],
        "name": "VoucherTransferred",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "voucherId", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "netAmount", "type": "uint256"},
            {"indexed": False, "internalType": "bool", "name": "isEth", "type": "bool"},
            {"indexed": False, "internalType": "address", "name": "tokenAddress", "type": "address"},
            {"indexed": False, "internalType": "address", "name": "recipient", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "redeemedAt", "type": "uint256"},
            {"indexed": False, "internalType": "address", "name": "depositor", "type": "address"},
        ],
        "name": "VoucherRedeemed",
        "type": "event",
    },
]

@dataclass
class Voucher:
    voucher_id: int
    amount: int
    secret_hash: str
    transfer_count: int
    redeemed: bool
    is_eth: bool
    token_address: str
    created_at: int
    last_transfer_at: int
    depositor: str
    redeemer: str
    cid: str
    chain: str

class VoucherSystem:
    KARMA_PER_VOUCHER_CREATION = 50
    KARMA_PER_VOUCHER_REDEMPTION = 20
    MIN_KARMA_TO_CREATE = 100

    def __init__(
        self,
        soulbound_system: SoulBoundIdentitySystem,
        yield_protocol: YieldProtocol,
        town_square: TownSquare,
        messaging: Messaging,
        consensus: Consensus,
        ipfs_endpoint: str = "/ip4/127.0.0.1/tcp/5001",
        log_level: str = "INFO",
    ):
        """Initialize VoucherSystem with dependencies."""
        self.logger = TCCLogger(level=log_level)
        self.security = BioInspiredSecuritySystem()
        self.security.establish_baseline()
        self.soulbound_system = soulbound_system
        self.yield_protocol = yield_protocol
        self.town_square = town_square
        self.messaging = messaging
        self.consensus = consensus
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

        self.w3_providers = {}
        self.contracts = {}
        for chain, config in CHAIN_CONFIGS.items():
            self.w3_providers[chain] = Web3(Web3.HTTPProvider(config["rpc_url"], request_kwargs={"timeout": 30}))
            if not self.w3_providers[chain].is_connected():
                self.logger.log(
                    "web3_init",
                    b"",
                    b"",
                    {"error": f"Web3 connection failed for {chain}"},
                    "ERROR",
                    "WEB3_INIT_FAILED"
                )
                raise RuntimeError(f"Web3 connection failed for {chain}")
            self.contracts[chain] = self.w3_providers[chain].eth.contract(
                address=config["contract_address"], abi=VOUCHER_ABI
            )

        self.admin_account = Account.from_key(ADMIN_PRIVATE_KEY)
        self.vouchers: Dict[int, Voucher] = {}
        self._load_vouchers()

    def _load_vouchers(self):
        """Load vouchers from data/vouchers.json."""
        try:
            with open("data/vouchers.json", "r", encoding="utf-8") as f:
                vouchers_data = json.load(f)
            for voucher_data in vouchers_data:
                self.vouchers[voucher_data["voucher_id"]] = Voucher(**voucher_data)
        except FileNotFoundError:
            self.logger.log("load_vouchers", b"", b"", {"info": "No vouchers file found, starting fresh"})
        except Exception as e:
            self.logger.log(
                "load_vouchers",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "LOAD_VOUCHERS_FAILED"
            )

    def _save_vouchers(self):
        """Save vouchers to data/vouchers.json."""
        try:
            with open("data/vouchers.json", "w", encoding="utf-8") as f:
                json.dump([vars(voucher) for voucher in self.vouchers.values()], f, indent=2)
        except Exception as e:
            self.logger.log(
                "save_vouchers",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "SAVE_VOUCHERS_FAILED"
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

    def _sign_transaction(self, tx: Dict, chain: str) -> Dict:
        """Sign a Web3 transaction with admin account."""
        try:
            signed_tx = self.w3_providers[chain].eth.account.sign_transaction(tx, self.admin_account.key)
            tx_hash = self.w3_providers[chain].eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3_providers[chain].eth.wait_for_transaction_receipt(tx_hash, timeout=120)
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

    def deposit_erc20(
        self, chain: str, token_address: str, amount: int, password_hash: str, depositor_address: str, signature: str
    ) -> int:
        """Deposit ERC20 tokens to create a voucher."""
        start_time = time.time_ns()
        try:
            # Verify identity and karma
            if not self.soulbound_system.verify_identity(depositor_address):
                raise ValueError("Invalid depositor identity")
            karma = self.yield_protocol.get_karma(depositor_address)
            if karma < self.MIN_KARMA_TO_CREATE:
                raise ValueError(f"Depositor karma {karma} below required {self.MIN_KARMA_TO_CREATE}")

            # Verify signature
            message = encode_defunct(text=f"deposit_erc20:{chain}:{token_address}:{amount}:{password_hash}")
            recovered = self.w3_providers[chain].eth.account.recover_message(message, signature=signature)
            if recovered.lower() != depositor_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            deposit_data = f"{chain}:{token_address}:{amount}:{password_hash}"
            is_anomaly, score, explanation = self.security.detect_anomaly(deposit_data.encode())
            if is_anomaly:
                self.logger.log(
                    "deposit_erc20_anomaly",
                    deposit_data.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Build transaction
            nonce = self.w3_providers[chain].eth.get_transaction_count(self.admin_account.address)
            tx = self.contracts[chain].functions.depositERC20(
                token_address, amount, password_hash
            ).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": self.w3_providers[chain].eth.gas_price,
                "chainId": CHAIN_CONFIGS[chain]["chain_id"],
            })
            receipt = self._sign_transaction(tx, chain)
            voucher_id = receipt.logs[0]["data"]  # Parse VoucherCreated event for voucherId

            # Store voucher metadata on IPFS
            voucher_data = {
                "voucher_id": voucher_id,
                "amount": amount,
                "secret_hash": password_hash,
                "transfer_count": 0,
                "redeemed": False,
                "is_eth": False,
                "token_address": token_address,
                "created_at": int(time.time()),
                "last_transfer_at": int(time.time()),
                "depositor": depositor_address,
                "redeemer": "0x0",
                "chain": chain,
            }
            cid = self._store_on_ipfs(json.dumps(voucher_data))

            # Create voucher object
            voucher = Voucher(**voucher_data, cid=cid)
            self.vouchers[voucher_id] = voucher
            self._save_vouchers()

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=depositor_address,
                amount=self.KARMA_PER_VOUCHER_CREATION,
                action_type="voucher_creation",
                signature=signature
            )

            # Log action
            self.logger.log(
                "deposit_erc20",
                json.dumps(voucher_data).encode(),
                cid.encode(),
                {
                    "voucher_id": voucher_id,
                    "depositor": depositor_address,
                    "chain": chain,
                    "karma_earned": self.KARMA_PER_VOUCHER_CREATION,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return voucher_id
        except Exception as e:
            self.logger.log(
                "deposit_erc20",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "DEPOSIT_ERC20_FAILED"
            )
            raise

    def deposit_eth(self, chain: str, amount: int, password_hash: str, depositor_address: str, signature: str) -> int:
        """Deposit ETH to create a voucher."""
        start_time = time.time_ns()
        try:
            # Verify identity and karma
            if not self.soulbound_system.verify_identity(depositor_address):
                raise ValueError("Invalid depositor identity")
            karma = self.yield_protocol.get_karma(depositor_address)
            if karma < self.MIN_KARMA_TO_CREATE:
                raise ValueError(f"Depositor karma {karma} below required {self.MIN_KARMA_TO_CREATE}")

            # Verify signature
            message = encode_defunct(text=f"deposit_eth:{chain}:{amount}:{password_hash}")
            recovered = self.w3_providers[chain].eth.account.recover_message(message, signature=signature)
            if recovered.lower() != depositor_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            deposit_data = f"{chain}:{amount}:{password_hash}"
            is_anomaly, score, explanation = self.security.detect_anomaly(deposit_data.encode())
            if is_anomaly:
                self.logger.log(
                    "deposit_eth_anomaly",
                    deposit_data.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Build transaction
            nonce = self.w3_providers[chain].eth.get_transaction_count(self.admin_account.address)
            tx = self.contracts[chain].functions.depositETH(password_hash).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "value": amount,
                "gas": 200000,
                "gasPrice": self.w3_providers[chain].eth.gas_price,
                "chainId": CHAIN_CONFIGS[chain]["chain_id"],
            })
            receipt = self._sign_transaction(tx, chain)
            voucher_id = receipt.logs[0]["data"]  # Parse VoucherCreated event for voucherId

            # Store voucher metadata on IPFS
            voucher_data = {
                "voucher_id": voucher_id,
                "amount": amount,
                "secret_hash": password_hash,
                "transfer_count": 0,
                "redeemed": False,
                "is_eth": True,
                "token_address": "0x0",
                "created_at": int(time.time()),
                "last_transfer_at": int(time.time()),
                "depositor": depositor_address,
                "redeemer": "0x0",
                "chain": chain,
            }
            cid = self._store_on_ipfs(json.dumps(voucher_data))

            # Create voucher object
            voucher = Voucher(**voucher_data, cid=cid)
            self.vouchers[voucher_id] = voucher
            self._save_vouchers()

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=depositor_address,
                amount=self.KARMA_PER_VOUCHER_CREATION,
                action_type="voucher_creation",
                signature=signature
            )

            # Log action
            self.logger.log(
                "deposit_eth",
                json.dumps(voucher_data).encode(),
                cid.encode(),
                {
                    "voucher_id": voucher_id,
                    "depositor": depositor_address,
                    "chain": chain,
                    "karma_earned": self.KARMA_PER_VOUCHER_CREATION,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return voucher_id
        except Exception as e:
            self.logger.log(
                "deposit_eth",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "DEPOSIT_ETH_FAILED"
            )
            raise

    def transfer_voucher(
        self, voucher_id: int, old_password: str, new_password_hash: str, sender_address: str, recipient_address: str, signature: str
    ) -> None:
        """Transfer a voucher by updating its password and notifying recipient."""
        start_time = time.time_ns()
        try:
            if voucher_id not in self.vouchers:
                raise ValueError("Voucher not found")
            voucher = self.vouchers[voucher_id]
            chain = voucher.chain

            # Verify identities
            if not self.soulbound_system.verify_identity(sender_address):
                raise ValueError("Invalid sender identity")
            if not self.soulbound_system.verify_identity(recipient_address):
                raise ValueError("Invalid recipient identity")

            # Verify signature
            message = encode_defunct(text=f"transfer_voucher:{voucher_id}:{old_password}:{new_password_hash}")
            recovered = self.w3_providers[chain].eth.account.recover_message(message, signature=signature)
            if recovered.lower() != sender_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            transfer_data = f"{voucher_id}:{old_password}:{new_password_hash}"
            is_anomaly, score, explanation = self.security.detect_anomaly(transfer_data.encode())
            if is_anomaly:
                self.logger.log(
                    "transfer_voucher_anomaly",
                    transfer_data.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Build transaction
            nonce = self.w3_providers[chain].eth.get_transaction_count(self.admin_account.address)
            tx = self.contracts[chain].functions.transferVoucher(
                voucher_id, old_password, new_password_hash
            ).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 150000,
                "gasPrice": self.w3_providers[chain].eth.gas_price,
                "chainId": CHAIN_CONFIGS[chain]["chain_id"],
            })
            receipt = self._sign_transaction(tx, chain)

            # Update voucher
            voucher.secret_hash = new_password_hash
            voucher.transfer_count += 1
            voucher.last_transfer_at = int(time.time())
            self._save_vouchers()

            # Notify recipient via Messaging
            conversation_id = self.messaging.create_conversation(
                initiator_address=sender_address,
                recipient_address=recipient_address,
                signature=signature
            )
            self.messaging.send_message(
                conversation_id=conversation_id,
                sender_address=sender_address,
                content=f"Voucher {voucher_id} transferred to you on {chain}.",
                signature=signature
            )

            # Log action
            self.logger.log(
                "transfer_voucher",
                transfer_data.encode(),
                b"",
                {
                    "voucher_id": voucher_id,
                    "sender": sender_address,
                    "recipient": recipient_address,
                    "chain": chain,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
        except Exception as e:
            self.logger.log(
                "transfer_voucher",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "TRANSFER_VOUCHER_FAILED"
            )
            raise

    def redeem_voucher(
        self, voucher_id: int, password: str, recipient_address: str, redeemer_address: str, signature: str
    ) -> None:
        """Redeem a voucher to transfer funds to recipient."""
        start_time = time.time_ns()
        try:
            if voucher_id not in self.vouchers:
                raise ValueError("Voucher not found")
            voucher = self.vouchers[voucher_id]
            chain = voucher.chain

            # Verify identities
            if not self.soulbound_system.verify_identity(redeemer_address):
                raise ValueError("Invalid redeemer identity")
            if not self.soulbound_system.verify_identity(recipient_address):
                raise ValueError("Invalid recipient identity")

            # Verify signature
            message = encode_defunct(text=f"redeem_voucher:{voucher_id}:{password}:{recipient_address}")
            recovered = self.w3_providers[chain].eth.account.recover_message(message, signature=signature)
            if recovered.lower() != redeemer_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            redeem_data = f"{voucher_id}:{password}:{recipient_address}"
            is_anomaly, score, explanation = self.security.detect_anomaly(redeem_data.encode())
            if is_anomaly:
                self.logger.log(
                    "redeem_voucher_anomaly",
                    redeem_data.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Build transaction
            nonce = self.w3_providers[chain].eth.get_transaction_count(self.admin_account.address)
            tx = self.contracts[chain].functions.redeemVoucher(
                voucher_id, password, recipient_address
            ).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 150000,
                "gasPrice": self.w3_providers[chain].eth.gas_price,
                "chainId": CHAIN_CONFIGS[chain]["chain_id"],
            })
            receipt = self._sign_transaction(tx, chain)

            # Update voucher
            voucher.redeemed = True
            voucher.redeemer = recipient_address
            self._save_vouchers()

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=redeemer_address,
                amount=self.KARMA_PER_VOUCHER_REDEMPTION,
                action_type="voucher_redemption",
                signature=signature
            )

            # Log action
            self.logger.log(
                "redeem_voucher",
                redeem_data.encode(),
                b"",
                {
                    "voucher_id": voucher_id,
                    "redeemer": redeemer_address,
                    "recipient": recipient_address,
                    "chain": chain,
                    "karma_earned": self.KARMA_PER_VOUCHER_REDEMPTION,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
        except Exception as e:
            self.logger.log(
                "redeem_voucher",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "REDEEM_VOUCHER_FAILED"
            )
            raise

    def port_cross_chain_voucher(
        self, source_chain: str, source_voucher_id: int, target_chain: str, amount: int, token_address: str,
        depositor_address: str, signature: str
    ) -> int:
        """Port a voucher from one chain to another, verified via consensus."""
        start_time = time.time_ns()
        try:
            # Verify identity and karma
            if not self.soulbound_system.verify_identity(depositor_address):
                raise ValueError("Invalid depositor identity")
            karma = self.yield_protocol.get_karma(depositor_address)
            if karma < self.MIN_KARMA_TO_CREATE:
                raise ValueError(f"Depositor karma {karma} below required {self.MIN_KARMA_TO_CREATE}")

            # Verify signature
            message = encode_defunct(
                text=f"port_voucher:{source_chain}:{source_voucher_id}:{target_chain}:{amount}:{token_address}"
            )
            recovered = self.w3_providers[target_chain].eth.account.recover_message(message, signature=signature)
            if recovered.lower() != depositor_address.lower():
                raise ValueError("Invalid signature")

            # Post claim to TownSquare for consensus
            post_content = (
                f"Request to port voucher {source_voucher_id} from {source_chain} to {target_chain} "
                f"with amount {amount} and token {token_address}."
            )
            post_id = self.town_square.create_post(
                content=post_content,
                author_address=depositor_address,
                signature=signature
            )

            # Start consensus validation
            claim_id = self.consensus.start_claim_validation(
                post_id=post_id,
                admin_address=depositor_address,
                signature=signature
            )

            # Assume consensus is reached (simplified for demo; in practice, wait for finalization)
            consensus_result = self.consensus.get_consensus_result(claim_id)
            if not consensus_result["is_finalized"] or not consensus_result["result"]:
                raise ValueError("Cross-chain porting not approved by consensus")

            # Create new voucher on target chain
            password_hash = self.w3_providers[target_chain].keccak(text=f"cross_chain:{source_voucher_id}").hex()
            voucher_id = self.deposit_erc20(
                chain=target_chain,
                token_address=token_address,
                amount=amount,
                password_hash=password_hash,
                depositor_address=depositor_address,
                signature=signature
            )

            # Notify via Messaging
            conversation_id = self.messaging.create_conversation(
                initiator_address=depositor_address,
                recipient_address=depositor_address,  # Self-notification for demo
                signature=signature
            )
            self.messaging.send_message(
                conversation_id=conversation_id,
                sender_address=depositor_address,
                content=f"Voucher {voucher_id} created on {target_chain} for cross-chain port from {source_chain}.",
                signature=signature
            )

            # Log action
            self.logger.log(
                "port_cross_chain_voucher",
                post_content.encode(),
                b"",
                {
                    "source_voucher_id": source_voucher_id,
                    "new_voucher_id": voucher_id,
                    "source_chain": source_chain,
                    "target_chain": target_chain,
                    "depositor": depositor_address,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return voucher_id
        except Exception as e:
            self.logger.log(
                "port_cross_chain_voucher",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "PORT_CROSS_CHAIN_VOUCHER_FAILED"
            )
            raise

    def get_voucher(self, voucher_id: int) -> Dict:
        """Retrieve voucher details."""
        start_time = time.time_ns()
        try:
            if voucher_id not in self.vouchers:
                raise ValueError("Voucher not found")
            voucher = self.vouchers[voucher_id]
            voucher_data = {
                "voucher_id": voucher.voucher_id,
                "amount": voucher.amount,
                "secret_hash": voucher.secret_hash,
                "transfer_count": voucher.transfer_count,
                "redeemed": voucher.redeemed,
                "is_eth": voucher.is_eth,
                "token_address": voucher.token_address,
                "created_at": voucher.created_at,
                "last_transfer_at": voucher.last_transfer_at,
                "depositor": voucher.depositor,
                "redeemer": voucher.redeemer,
                "cid": voucher.cid,
                "chain": voucher.chain,
            }
            self.logger.log(
                "get_voucher",
                f"voucher_id:{voucher_id}".encode(),
                json.dumps(voucher_data).encode(),
                {"execution_time_ns": time.time_ns() - start_time}
            )
            return voucher_data
        except Exception as e:
            self.logger.log(
                "get_voucher",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "GET_VOUCHER_FAILED"
            )
            raise