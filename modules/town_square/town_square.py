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
from .messaging import Messaging
from .consensus import Consensus
from .voucher_system import VoucherSystem
from ..agents.tcc_logger import TCCLogger
import ipfshttpclient

# Configure logging
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("onoclast_town_square")

# Environment variables
ARBITRUM_RPC_URL = "https://arb1.arbitrum.io/rpc"
TOWN_SQUARE_CONTRACT_ADDRESS = Web3.to_checksum_address("0xYourTownSquareContractAddress")  # Placeholder
ADMIN_PRIVATE_KEY = "0xYourAdminPrivateKey"  # Placeholder, store securely

# TownSquare Contract ABI (simplified)
TOWN_SQUARE_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "contentCid", "type": "string"},
            {"internalType": "address", "name": "author", "type": "address"},
            {"internalType": "uint256", "name": "voucherId", "type": "uint256"},
        ],
        "name": "createPost",
        "outputs": [{"internalType": "uint256", "name": "postId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "postId", "type": "uint256"},
            {"internalType": "string", "name": "contentCid", "type": "string"},
            {"internalType": "address", "name": "author", "type": "address"},
            {"internalType": "uint256", "name": "voucherId", "type": "uint256"},
        ],
        "name": "createComment",
        "outputs": [{"internalType": "uint256", "name": "commentId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "postId", "type": "uint256"},
            {"internalType": "address", "name": "voter", "type": "address"},
            {"internalType": "bool", "name": "isUpvote", "type": "bool"},
        ],
        "name": "votePost",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

@dataclass
class Comment:
    comment_id: int
    post_id: int
    author: str
    content: str
    timestamp: int
    voucher_id: int
    cid: str
    upvotes: int = 0
    downvotes: int = 0

@dataclass
class Post:
    post_id: int
    author: str
    content: str
    timestamp: int
    voucher_id: int
    cid: str
    comments: List[Comment] = field(default_factory=list)
    upvotes: int = 0
    downvotes: int = 0

class TownSquare:
    KARMA_PER_POST = 100
    KARMA_PER_COMMENT = 50
    KARMA_PER_UPVOTE = 10
    MIN_KARMA_TO_POST = 200
    MIN_KARMA_TO_COMMENT = 100

    def __init__(
        self,
        soulbound_system: SoulBoundIdentitySystem,
        yield_protocol: YieldProtocol,
        messaging: Messaging,
        consensus: Consensus,
        voucher_system: VoucherSystem,
        ipfs_endpoint: str = "/ip4/127.0.0.1/tcp/5001",
        log_level: str = "INFO",
    ):
        """Initialize TownSquare with dependencies."""
        self.logger = TCCLogger(level=log_level)
        self.security = BioInspiredSecuritySystem()
        self.security.establish_baseline()
        self.soulbound_system = soulbound_system
        self.yield_protocol = yield_protocol
        self.messaging = messaging
        self.consensus = consensus
        self.voucher_system = voucher_system
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

        self.contract = self.w3.eth.contract(address=TOWN_SQUARE_CONTRACT_ADDRESS, abi=TOWN_SQUARE_ABI)
        self.admin_account = Account.from_key(ADMIN_PRIVATE_KEY)
        self.posts: Dict[int, Post] = {}
        self._load_content()

    def _load_content(self):
        """Load posts and comments from data/town_square.json."""
        try:
            with open("data/town_square.json", "r", encoding="utf-8") as f:
                content_data = json.load(f)
            for post_data in content_data.get("posts", []):
                post_data["comments"] = [Comment(**comment) for comment in post_data.get("comments", [])]
                self.posts[post_data["post_id"]] = Post(**post_data)
        except FileNotFoundError:
            self.logger.log("load_content", b"", b"", {"info": "No content file found, starting fresh"})
        except Exception as e:
            self.logger.log(
                "load_content",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "LOAD_CONTENT_FAILED"
            )

    def _save_content(self):
        """Save posts and comments to data/town_square.json."""
        try:
            with open("data/town_square.json", "w", encoding="utf-8") as f:
                json.dump({
                    "posts": [vars(post) for post in self.posts.values()]
                }, f, indent=2)
        except Exception as e:
            self.logger.log(
                "save_content",
                b"",
                b"",
                {"error": str(e)},
                "ERROR",
                "SAVE_CONTENT_FAILED"
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

    def create_post(self, content: str, author_address: str, voucher_id: Optional[int], signature: str) -> int:
        """Create a new post, optionally with a voucher."""
        start_time = time.time_ns()
        try:
            # Verify identity and karma
            if not self.soulbound_system.verify_identity(author_address):
                raise ValueError("Invalid author identity")
            karma = self.yield_protocol.get_karma(author_address)
            if karma < self.MIN_KARMA_TO_POST:
                raise ValueError(f"Author karma {karma} below required {self.MIN_KARMA_TO_POST}")

            # Verify signature
            message = encode_defunct(text=f"create_post:{content}:{voucher_id or 0}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != author_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            is_anomaly, score, explanation = self.security.detect_anomaly(content.encode())
            if is_anomaly:
                self.logger.log(
                    "create_post_anomaly",
                    content.encode(),
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
                if voucher["depositor"] != author_address:
                    raise ValueError("Voucher not owned by author")

            # Store content on IPFS
            post_data = {
                "content": content,
                "author": author_address,
                "voucher_id": voucher_id or 0,
                "timestamp": int(time.time()),
            }
            cid = self._store_on_ipfs(json.dumps(post_data))

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.createPost(cid, author_address, voucher_id or 0).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,  # Arbitrum One
            })
            receipt = self._sign_transaction(tx)
            post_id = receipt.logs[0]["data"]  # Parse PostCreated event for postId

            # Create post object
            post = Post(
                post_id=post_id,
                author=author_address,
                content=content,
                timestamp=int(time.time()),
                voucher_id=voucher_id or 0,
                cid=cid,
            )
            self.posts[post_id] = post
            self._save_content()

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=author_address,
                amount=self.KARMA_PER_POST,
                action_type="post_creation",
                signature=signature
            )

            # Log action
            self.logger.log(
                "create_post",
                json.dumps(post_data).encode(),
                cid.encode(),
                {
                    "post_id": post_id,
                    "author": author_address,
                    "voucher_id": voucher_id or 0,
                    "karma_earned": self.KARMA_PER_POST,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return post_id
        except Exception as e:
            self.logger.log(
                "create_post",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "CREATE_POST_FAILED"
            )
            raise

    def create_comment(self, post_id: int, content: str, author_address: str, voucher_id: Optional[int], signature: str) -> int:
        """Create a comment on a post, optionally with a voucher."""
        start_time = time.time_ns()
        try:
            if post_id not in self.posts:
                raise ValueError("Post not found")
            # Verify identity and karma
            if not self.soulbound_system.verify_identity(author_address):
                raise ValueError("Invalid author identity")
            karma = self.yield_protocol.get_karma(author_address)
            if karma < self.MIN_KARMA_TO_COMMENT:
                raise ValueError(f"Author karma {karma} below required {self.MIN_KARMA_TO_COMMENT}")

            # Verify signature
            message = encode_defunct(text=f"create_comment:{post_id}:{content}:{voucher_id or 0}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != author_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            is_anomaly, score, explanation = self.security.detect_anomaly(content.encode())
            if is_anomaly:
                self.logger.log(
                    "create_comment_anomaly",
                    content.encode(),
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
                if voucher["depositor"] != author_address:
                    raise ValueError("Voucher not owned by author")

            # Store content on IPFS
            comment_data = {
                "post_id": post_id,
                "content": content,
                "author": author_address,
                "voucher_id": voucher_id or 0,
                "timestamp": int(time.time()),
            }
            cid = self._store_on_ipfs(json.dumps(comment_data))

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.createComment(post_id, cid, author_address, voucher_id or 0).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            receipt = self._sign_transaction(tx)
            comment_id = receipt.logs[0]["data"]  # Parse CommentCreated event for commentId

            # Create comment object
            comment = Comment(
                comment_id=comment_id,
                post_id=post_id,
                author=author_address,
                content=content,
                timestamp=int(time.time()),
                voucher_id=voucher_id or 0,
                cid=cid,
            )
            self.posts[post_id].comments.append(comment)
            self._save_content()

            # Notify post author via Messaging
            post = self.posts[post_id]
            if post.author != author_address:
                conversation_id = self.messaging.create_conversation(
                    initiator_address=author_address,
                    recipient_address=post.author,
                    signature=signature
                )
                self.messaging.send_message(
                    conversation_id=conversation_id,
                    sender_address=author_address,
                    content=f"New comment on your post {post_id}: {content[:50]}...",
                    signature=signature
                )

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=author_address,
                amount=self.KARMA_PER_COMMENT,
                action_type="comment_creation",
                signature=signature
            )

            # Log action
            self.logger.log(
                "create_comment",
                json.dumps(comment_data).encode(),
                cid.encode(),
                {
                    "comment_id": comment_id,
                    "post_id": post_id,
                    "author": author_address,
                    "voucher_id": voucher_id or 0,
                    "karma_earned": self.KARMA_PER_COMMENT,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return comment_id
        except Exception as e:
            self.logger.log(
                "create_comment",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "CREATE_COMMENT_FAILED"
            )
            raise

    def vote_post(self, post_id: int, voter_address: str, is_upvote: bool, signature: str) -> None:
        """Vote on a post (upvote or downvote)."""
        start_time = time.time_ns()
        try:
            if post_id not in self.posts:
                raise ValueError("Post not found")
            # Verify identity
            if not self.soulbound_system.verify_identity(voter_address):
                raise ValueError("Invalid voter identity")

            # Verify signature
            message = encode_defunct(text=f"vote_post:{post_id}:{is_upvote}")
            recovered = self.w3.eth.account.recover_message(message, signature=signature)
            if recovered.lower() != voter_address.lower():
                raise ValueError("Invalid signature")

            # Security check
            vote_data = f"{post_id}:{voter_address}:{is_upvote}"
            is_anomaly, score, explanation = self.security.detect_anomaly(vote_data.encode())
            if is_anomaly:
                self.logger.log(
                    "vote_post_anomaly",
                    vote_data.encode(),
                    b"",
                    {"score": score, "explanation": explanation},
                    "ERROR",
                    "ANOMALY_DETECTED"
                )
                raise ValueError(f"Anomaly detected: {explanation}")

            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.admin_account.address)
            tx = self.contract.functions.votePost(post_id, voter_address, is_upvote).build_transaction({
                "from": self.admin_account.address,
                "nonce": nonce,
                "gas": 150000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 42161,
            })
            self._sign_transaction(tx)

            # Update post
            post = self.posts[post_id]
            if is_upvote:
                post.upvotes += 1
            else:
                post.downvotes += 1
            self._save_content()

            # Notify post author via Messaging
            if post.author != voter_address:
                conversation_id = self.messaging.create_conversation(
                    initiator_address=voter_address,
                    recipient_address=post.author,
                    signature=signature
                )
                self.messaging.send_message(
                    conversation_id=conversation_id,
                    sender_address=voter_address,
                    content=f"Your post {post_id} received an {'upvote' if is_upvote else 'downvote'}.",
                    signature=signature
                )

            # Accrue karma
            self.yield_protocol.accrue_karma(
                account=voter_address,
                amount=self.KARMA_PER_UPVOTE,
                action_type="vote",
                signature=signature
            )

            # Log action
            self.logger.log(
                "vote_post",
                vote_data.encode(),
                b"",
                {
                    "post_id": post_id,
                    "voter": voter_address,
                    "is_upvote": is_upvote,
                    "karma_earned": self.KARMA_PER_UPVOTE,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
        except Exception as e:
            self.logger.log(
                "vote_post",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "VOTE_POST_FAILED"
            )
            raise

    def get_post(self, post_id: int) -> Dict:
        """Retrieve a post and its comments."""
        start_time = time.time_ns()
        try:
            if post_id not in self.posts:
                raise ValueError("Post not found")
            post = self.posts[post_id]
            post_data = {
                "post_id": post.post_id,
                "author": post.author,
                "content": post.content,
                "timestamp": post.timestamp,
                "voucher_id": post.voucher_id,
                "cid": post.cid,
                "upvotes": post.upvotes,
                "downvotes": post.downvotes,
                "comments": [
                    {
                        "comment_id": comment.comment_id,
                        "post_id": comment.post_id,
                        "author": comment.author,
                        "content": comment.content,
                        "timestamp": comment.timestamp,
                        "voucher_id": comment.voucher_id,
                        "cid": comment.cid,
                        "upvotes": comment.upvotes,
                        "downvotes": comment.downvotes,
                    }
                    for comment in post.comments
                ],
            }
            self.logger.log(
                "get_post",
                f"post_id:{post_id}".encode(),
                json.dumps(post_data).encode(),
                {"execution_time_ns": time.time_ns() - start_time}
            )
            return post_data
        except Exception as e:
            self.logger.log(
                "get_post",
                b"",
                b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time},
                "ERROR",
                "GET_POST_FAILED"
            )
            raise