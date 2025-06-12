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
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import os
from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
from .security import BioInspiredSecuritySystem
from .soulbound_identity import SoulBoundIdentitySystem
from .yield_protocol import YieldProtocol
from ..agents.tcc_logger import TCCLogger

# Configure logging
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("onoclast_messaging")

# Environment variables
ARBITRUM_RPC_URL = "https://arb1.arbitrum.io/rpc"
MESSAGING_CONTRACT_ADDRESS = Web3.to_checksum_address("0xYourMessagingContractAddress")  # Placeholder
ADMIN_PRIVATE_KEY = "0xYourAdminPrivateKey"  # Placeholder, store securely

# Messaging Contract ABI (simplified)
MESSAGING_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "conversationCid", "type": "string"},
            {"internalType": "address", "name": "initiator", "type": "address"},
            {"internalType": "address", "name": "recipient", "type": "address"},
        ],
        "name": "createConversation",
        "outputs": [{"internalType": "uint256", "name": "conversationId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "string", "name": "groupCid", "type": "string"},
            {"internalType": "address", "name": "admin", "type": "address"},
        ],
        "name": "createGroup",
        "outputs": [{"internalType": "uint256", "name": "groupId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "groupId", "type": "uint256"},
            {"internalType": "address", "name": "member", "type": "address"},
        ],
        "name": "addGroupMember",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

@dataclass
class Message:
    sender: str
    content: bytes  # Encrypted content
    timestamp: int
    cid: str

@dataclass
class Conversation:
    conversation_id: int
    initiator: str
    recipient: str
    messages: List[Message] = field(default_factory=list)
    cid: str = ""

@dataclass
class Group:
    group_id: int
    admin: str
    members: List[str]
    messages: List[Message] = field(default_factory=list)
    cid: str = ""
    min_karma: int = 100  # Minimum karma to join

class Messaging:
    KARMA_PER_CONVERSATION = 10
    KARMA_PER_GROUP_CREATION = 50
    KARMA_PER_MESSAGE = 5

    def __init__(self, soulbound_system: SoulBoundIdentitySystem, yield_protocol: YieldProtocol, ipfs_endpoint: str = "/ip4/127.0.0.1/tcp/5001", log_level: str = "INFO"):
        """Initialize Messaging with IPFS, Web3, security, and logging."""
        self.logger = TCCLogger(level=log_level)
        self.security = BioInspiredSecuritySystem()
        self.security.establish_baseline()
        self.soulbound_system = soulbound_system
        self.yield_protocol = yield_protocol
        try:
            import ipfshttpclient
            self.ipfs_client = ipfshttpclient.connect(ipfs_endpoint)
        except Exception as e:
            self.logger.log(
                "ipfs_init",
                b"",
                "",
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
                "",
                {"error": "Web3 connection failed"},
                "ERROR",
                "WEB3_INIT_FAILED"
            )
            raise RuntimeError("Web3 connection failed")

        self.contract = self.w3.eth.contract(address=MESSAGING_CONTRACT_ADDRESS, abi=MESSAGING_ABI)
        self.admin_account = Account.from_key(ADMIN_PRIVATE_KEY)
        self.conversations: Dict[int, Conversation] = {}
        self.groups: Dict[int, Group] = {}
        self._load_messages()

    def _load_messages(self):
        """Load conversations and groups from data/messages.json."""
        try:
            with open("data/messages.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            for conv_data in data.get("conversations", []):
                self.conversations[conv_data["conversation_id"]] = Conversation(**conv_data)
            for group_data in data.get("groups", []):
                self.groups[group_data["group_id"]] = Group(**group_data)
        except FileNotFoundError:
            self.logger.log("load_messages", b"", {"info": "No messages file found, starting fresh"})
        except Exception as e:
            self.logger.log(
                "load_messages",
                b"",
                {"error": str(e)},
                "ERROR",
                "LOAD_MESSAGES_FAILED"
            )

    def _save_messages(self):
        """Save conversations and groups to data/messages.json."""
        try:
            with open("data/messages.json", "w", encoding="utf-8") as f:
                json.dump({
                    "conversations": [vars(conv) for conv in self.conversations.values()],
                    "groups": [vars(group) for group in self.groups.values()],
                }, f, indent=4)
        except Exception as e:
            self.logger.log(
                "save_messages",
                b"",
                {"error": str(e)},
                "ERROR",
                "SAVE_MESSAGES_FAILED"
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
                "IPFS_REQUEST_FAILED"
            )
            raise

    def _encrypt_message(self, content: str, recipient_address: str) -> bytes:
        """Encrypt a message using recipient's public key."""
        try:
            # Get recipient's public key from SoulBoundIdentity
            identity = self.soulbound_system.get_identity(recipient_address)
            public_key_bytes = base64.b64decode(identity.metadata["public_key"])
            public_key = serialization.load_der_public_key(public_key_bytes)

            # Generate shared key using sender's private key (admin for demo)
            private_key = ec.generate_private_key(ec.SECP256K1())
            shared_key = private_key.exchange(ec.ECDH(), public_key)

            # Derive AES key
            aes_key = hashes.Hash(hashes.SHA256())
            aes_key.update(shared_key)
            aes_key = aes_key.finalize()[:32]

            # Encrypt with AES
            nonce = os.urandom(12)
            cipher = Cipher(algorithms.AES(aes_key), modes.GCM(nonce))
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(content.encode()) + encryptor.finalize()
            return base64.b64encode(nonce + ciphertext + encryptor.tag)
        except Exception as e:
            self.logger.log(
                "encrypt_message",
                b"",
                {"error": str(e)},
                "ERROR",
                "ENCRYPT_MESSAGE_FAILED"
            )
            raise

    def _sign_transaction(self, tx: Dict) -> Dict:
        """Sign a Web3 transaction with admin account."""
        try:
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.admin_account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120))
            if receipt.status != 1:
                raise ValueError("Transaction failed")
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

    def create_conversation(self, initiator_address: str, recipient_address: str, signature: str) -> int:
        """Create a new conversation between two users, rewarding initiator."""
        start_time = time.time_ns()
        try:
            # Verify identities
            if not self.soulbound_system.verify_identity(initiator_address):
                raise ValueError("Invalid initiator identity")
            if not self.soulbound_system.verify_identity(recipient_address):
                raise ValueError("Invalid recipient identity")

            # Verify signature
            message = encode_defunct(text=f"conversation:{initiator_address}:{recipient_address}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != initiator_address.lower():
                raise ValueError("Invalid signature")

            # Store conversation metadata on IPFS
            conv_data = {
                "initiator": initiator_address,
                "recipient": recipient_address,
                "timestamp": int(time.time()),
            }
            cid = self._store_on_ipfs(json.dumps(conv_data))

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.createConversation(cid, initiator_address, recipient_address).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,  # Arbitrum One
            })
            receipt = self._sign_transaction(tx)
            conversation_id = self.contract.events.ConversationCreated().process_receipt(receipt)[0]["args"]["conversationId"]

            # Create conversation object
            conversation = Conversation(
                conversation_id=conversation_id,
                initiator=initiator_address,
                recipient=recipient_address,
                cid=cid,
            )
            self.conversations[conversation_id] = conversation
            self._save_messages()

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=initiator_address,
                amount=self.KARMA_PER_CONVERSATION,
                action_type="conversation",
                signature=signature
            )

            # Log action
            self.logger.log(
                "create_conversation",
                json.dumps(conv_data).encode(),
                cid.encode(),
                {
                    "conversation_id": conversation_id,
                    "initiator": initiator_address,
                    "recipient": recipient_address,
                    "karma_earned": self.KARMA_PER_CONVERSATION,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return conversation_id
        except Exception as e:
            self.logger.log(
                "create_conversation",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "CREATE_CONVERSATION_FAILED"
            )
            raise

    def send_message(self, conversation_id: int, sender_address: str, content: str, signature: str) -> None:
        """Send an encrypted message in a conversation."""
        start_time = time.time_ns()
        try:
            if conversation_id not in self.conversations:
                raise ValueError("Conversation not found")
            conversation = self.conversations[conversation_id]
            if sender_address not in (conversation.initiator, conversation.recipient):
                raise ValueError("Sender not part of conversation")

            # Verify identity and signature
            if not self.soulbound_system.verify_identity(sender_address):
                raise ValueError("Invalid sender identity")
            message = encode_defunct(text=content)
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != sender_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            is_anomaly, score, explanation = self.security.detect_anomaly(content.encode())
            if is_anomaly:
                self.logger.log(
                    "send_message_anomaly",
                    content.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Encrypt message for recipient
            recipient = conversation.recipient if sender_address == conversation.initiator else conversation.initiator
            encrypted_content = self._encrypt_message(content, recipient)

            # Store on IPFS
            message_data = {
                "sender": sender_address,
                "content": base64.b64encode(encrypted_content).decode(),
                "timestamp": int(time.time()),
            }
            cid = self._store_on_ipfs(json.dumps(message_data))

            # Add message to conversation
            message = Message(
                sender=sender_address,
                content=encrypted_content,
                timestamp=int(time.time()),
                cid=cid,
            )
            conversation.messages.append(message)
            self._save_messages()

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=sender_address,
                amount=self.KARMA_PER_MESSAGE,
                action_type="message",
                signature=signature
            )

            # Log action
            self.logger.log(
                "send_message",
                content.encode(),
                cid.encode(),
                {
                    "conversation_id": conversation_id,
                    "sender": sender_address,
                    "karma_earned": self.KARMA_PER_MESSAGE,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
        except Exception as e:
            self.logger.log(
                "send_message",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "SEND_MESSAGE_FAILED"
            )
            raise

    def create_group(self, admin_address: str, min_karma: int, signature: str) -> int:
        """Create a new group with karma-based membership, rewarding admin."""
        start_time = time.time_ns()
        try:
            # Verify identity and karma
            if not self.soulbound_system.verify_identity(admin_address):
                raise ValueError("Invalid admin identity")
            karma = self.yield_protocol.get_karma(admin_address)
            if karma < min_karma:
                raise ValueError(f"Admin karma {karma} below required {min_karma}")

            # Verify signature
            message = encode_defunct(text=f"group:{admin_address}:{min_karma}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != admin_address.lower():
                raise ValueError("Invalid signature")

            # Store group metadata on IPFS
            group_data = {
                "admin": admin_address,
                "min_karma": min_karma,
                "timestamp": int(time.time()),
            }
            cid = self._store_on_ipfs(json.dumps(group_data))

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.createGroup(cid, admin_address).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            receipt = self._sign_transaction(tx)
            group_id = self.contract.events.GroupCreated().process_receipt(receipt)[0]["args"]["groupId"]

            # Create group object
            group = Group(
                group_id=group_id,
                admin=admin_address,
                members=[admin_address],
                cid=cid,
                min_karma=min_karma,
            )
            self.groups[group_id] = group
            self._save_messages()

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=admin_address,
                amount=self.KARMA_PER_GROUP_CREATION,
                action_type="group_creation",
                signature=signature
            )

            # Log action
            self.logger.log(
                "create_group",
                json.dumps(group_data).encode(),
                cid.encode(),
                {
                    "group_id": group_id,
                    "admin": admin_address,
                    "min_karma": min_karma,
                    "karma_earned": self.KARMA_PER_GROUP_CREATION,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return group_id
        except Exception as e:
            self.logger.log(
                "create_group",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "CREATE_GROUP_FAILED"
            )
            raise

    def add_group_member(self, group_id: int, member_address: str, admin_address: str, signature: str) -> None:
        """Add a member to a group if they meet karma requirements."""
        start_time = time.time_ns()
        try:
            if group_id not in self.groups:
                raise ValueError("Group not found")
            group = self.groups[group_id]
            if admin_address != group.admin:
                raise ValueError("Only admin can add members")

            # Verify identities and karma
            if not self.soulbound_system.verify_identity(member_address):
                raise ValueError("Invalid member identity")
            if not self.soulbound_system.verify_identity(admin_address):
                raise ValueError("Invalid admin identity")
            karma = self.yield_protocol.get_karma(member_address)
            if karma < group.min_karma:
                raise ValueError(f"Member karma {karma} below required {group.min_karma}")

            # Verify signature
            message = encode_defunct(text=f"add_member:{group_id}:{member_address}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != admin_address.lower():
                raise ValueError("Invalid signature")

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.addGroupMember(group_id, member_address).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 150000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            self._sign_transaction(tx)

            # Update group
            if member_address not in group.members:
                group.members.append(member_address)
                self._save_messages()

            # Log action
            self.logger.log(
                "add_group_member",
                f"group_id:{group_id}".encode(),
                b"",
                {
                    "group_id": group_id,
                    "member": member_address,
                    "admin": admin_address,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
        except Exception as e:
            self.logger.log(
                "add_group_member",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "ADD_GROUP_MEMBER_FAILED"
            )
            raise

    def send_group_message(self, group_id: int, sender_address: str, content: str, signature: str) -> None:
        """Send an encrypted message to a group."""
        start_time = time.time_ns()
        try:
            if group_id not in self.groups:
                raise ValueError("Group not found")
            group = self.groups[group_id]
            if sender_address not in group.members:
                raise ValueError("Sender not a group member")

            # Verify identity and signature
            if not self.soulbound_system.verify_identity(sender_address):
                raise ValueError("Invalid sender identity")
            message = encode_defunct(text=content)
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != sender_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            is_anomaly, score, explanation = self.security.detect_anomaly(content.encode())
            if is_anomaly:
                self.logger.log(
                    "send_group_message_anomaly",
                    content.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Encrypt message for each member
            encrypted_contents = {}
            for member in group.members:
                encrypted_content = self._encrypt_message(content, member)
                encrypted_contents[member] = base64.b64encode(encrypted_content).decode()

            # Store on IPFS
            message_data = {
                "sender": sender_address,
                "contents": encrypted_contents,
                "timestamp": int(time.time()),
            }
            cid = self._store_on_ipfs(json.dumps(message_data))

            # Add message to group
            message = Message(
                sender=sender_address,
                content=encrypted_contents[sender_address].encode(),  # Store sender's copy
                timestamp=int(time.time()),
                cid=cid,
            )
            group.messages.append(message)
            self._save_messages()

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=sender_address,
                amount=self.KARMA_PER_MESSAGE,
                action_type="group_message",
                signature=signature
            )

            # Log action
            self.logger.log(
                "send_group_message",
                content.encode(),
                cid.encode(),
                {
                    "group_id": group_id,
                    "sender": sender_address,
                    "karma_earned": self.KARMA_PER_MESSAGE,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
        except Exception as e:
            self.logger.log(
                "send_group_message",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "SEND_GROUP_MESSAGE_FAILED"
            )
            raise

    def get_conversation(self, conversation_id: int) -> Dict:
        """Retrieve a conversation and its messages."""
        start_time = time.time_ns()
        try:
            if conversation_id not in self.conversations:
                raise ValueError("Conversation not found")
            conv = self.conversations[conversation_id]
            conv_data = {
                "conversation_id": conv.conversation_id,
                "initiator": conv.initiator,
                "recipient": conv.recipient,
                "cid": conv.cid,
                "messages": [
                    {
                        "sender": msg.sender,
                        "content": base64.b64encode(msg.content).decode(),
                        "timestamp": msg.timestamp,
                        "cid": msg.cid,
                    }
                    for msg in conv.messages
                ],
            }
            self.logger.log(
                "get_conversation",
                f"conversation_id:{conversation_id}".encode(),
                json.dumps(conv_data).encode(),
                {"execution_time_ns": time.time_ns() - start_time}
            )
            return conv_data
        except Exception as e:
            self.logger.log(
                "get_conversation",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "GET_CONVERSATION_FAILED"
            )
            raise

    def get_group(self, group_id: int) -> Dict:
        """Retrieve a group and its messages."""
        start_time = time.time_ns()
        try:
            if group_id not in self.groups:
                raise ValueError("Group not found")
            group = self.groups[group_id]
            group_data = {
                "group_id": group.group_id,
                "admin": group.admin,
                "members": group.members,
                "min_karma": group.min_karma,
                "cid": group.cid,
                "messages": [
                    {
                        "sender": msg.sender,
                        "content": base64.b64encode(msg.content).decode(),
                        "timestamp": msg.timestamp,
                        "cid": msg.cid,
                    }
                    for msg in group.messages
                ],
            }
            self.logger.log(
                "get_group",
                f"group_id:{group_id}".encode(),
                json.dumps(group_data).encode(),
                {"execution_time_ns": time.time_ns() - start_time}
            )
            return group_data
        except Exception as e:
            self.logger.log(
                "get_group",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "GET_GROUP_FAILED"
            )
            raise