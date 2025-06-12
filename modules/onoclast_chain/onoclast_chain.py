import time
import json
import hashlib
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from eth_account import Account
from eth_account.messages import encode_defunct
import ipfshttpclient
import libp2p
from .consensus import Consensus
from .fractal_pulse import PulseSystem
from .town_square import TownSquare
from .sovereign_script_interpreter import SovereignScriptInterpreter
from .security import BioInspiredSecuritySystem
from .soulbound_identity import SoulBoundIdentitySystem
from .yield_protocol import YieldProtocol
from .messaging import Messaging
from .voucher_system import VoucherSystem
from ..agents.tcc_logger import TCCLogger
import os

@dataclass
class Transaction:
    tx_id: str
    sender: str
    type: str
    data: Dict
    timestamp: int
    signature: str

@dataclass
class Block:
    block_height: int
    previous_hash: str
    transactions: List[Transaction]
    timestamp: int
    validator: str
    signature: str
    block_hash: str

class OnoclastChain:
    GENESIS_HASH = "0" * 64
    BLOCK_INTERVAL = 10
    MIN_KARMA_TO_VALIDATE = 200

    def __init__(
        self,
        node_id: str,
        user_address: str,
        private_key: str,
        ipfs_endpoint: str = "/ip4/127.0.0.1/tcp/5001",
        libp2p_endpoint: str = "/ip4/127.0.0.1/tcp/4001",
        state_dir: str = "data/chain_state/",
        log_level: str = "INFO"
    ):
        self.logger = TCCLogger(level=log_level)
        self.node_id = node_id
        self.user_address = user_address
        self.account = Account.from_key(private_key)
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        
        self.security = BioInspiredSecuritySystem()
        self.yield_protocol = YieldProtocol()
        self.messaging = Messaging()
        
        self.soulbound_system = SoulBoundIdentitySystem(
            yield_protocol=self.yield_protocol,
            ipfs_endpoint=ipfs_endpoint,
            log_level=log_level,
            p2p=None
        )
        
        self.voucher_system = VoucherSystem(
            soulbound_system=self.soulbound_system,
            yield_protocol=self.yield_protocol,
            town_square=None,
            messaging=self.messaging,
            consensus=None,
            ipfs_endpoint=ipfs_endpoint,
            log_level=log_level
        )
        
        self.town_square = TownSquare(
            soulbound_system=self.soulbound_system,
            yield_protocol=self.yield_protocol,
            messaging=self.messaging,
            consensus=None,
            voucher_system=self.voucher_system,
            node_id=node_id,
            private_key=private_key,
            ipfs_endpoint=ipfs_endpoint,
            log_level=log_level
        )
        
        self.consensus = Consensus(
            soulbound_system=self.soulbound_system,
            yield_protocol=self.yield_protocol,
            town_square=self.town_square,
            messaging=self.messaging,
            ipfs_endpoint=ipfs_endpoint,
            log_level=log_level
        )
        self.town_square.consensus = self.consensus
        self.voucher_system.consensus = self.consensus
        self.voucher_system.town_square = self.town_square
        
        self.pulse_system = PulseSystem(
            soulbound_system=self.soulbound_system,
            yield_protocol=self.yield_protocol,
            messaging=self.messaging,
            consensus=self.consensus,
            voucher_system=self.voucher_system,
            town_square=self.town_square,
            security=self.security,
            state_file=f"{state_dir}{node_id}_pulse_state.json",
            ledger_file=f"{state_dir}{node_id}_pulse_ledger.json",
            log_dir=f"{state_dir}logs/"
        )
        
        self.script_interpreter = SovereignScriptInterpreter(
            town_square=self.town_square,
            soulbound_system=self.soulbound_system,
            yield_protocol=self.yield_protocol,
            user_address=user_address,
            private_key=private_key,
            log_level=log_level
        )
        
        try:
            self.ipfs_client = ipfshttpclient.connect(ipfs_endpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to IPFS: {e}")
        try:
            self.p2p = libp2p.new_node(multiaddr=libp2p_endpoint)
            self.p2p.set_stream_handler("/onoclast/1.0.0", self._handle_gossip)
            self.soulbound_system.p2p = self.p2p
            self.voucher_system.p2p = self.p2p
            self.consensus.p2p = self.p2p
        except Exception as e:
            raise RuntimeError(f"Failed to connect to libp2p: {e}")
        
        self.blocks: List[Block] = []
        self.pending_txs: List[Transaction] = []
        self.chain_state: Dict = {"accounts": {}, "claims": {}, "pulses": {}, "posts": {}, "identities": {}, "vouchers": {}}
        self._load_chain_state()

    def _load_chain_state(self):
        try:
            with open(f"{self.state_dir}{self.node_id}_chain.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.blocks = [Block(**block) for block in data.get("blocks", [])]
                self.chain_state = data.get("state", {"accounts": {}, "claims": {}, "pulses": {}, "posts": {}, "identities": {}, "vouchers": {}})
        except FileNotFoundError:
            self._create_genesis_block()

    def _save_chain_state(self):
        try:
            with open(f"{self.state_dir}{self.node_id}_chain.json", "w", encoding="utf-8") as f:
                json.dump({"blocks": [vars(block) for block in self.blocks], "state": self.chain_state}, f, indent=2)
        except Exception as e:
            self.logger.log("save_chain_state", b"", b"", {"error": str(e)}, "ERROR", "SAVE_CHAIN_STATE_FAILED")

    def _create_genesis_block(self):
        genesis_block = Block(
            block_height=0,
            previous_hash=self.GENESIS_HASH,
            transactions=[],
            timestamp=int(time.time()),
            validator=self.user_address,
            signature=self._sign_message(f"genesis:{self.user_address}:0"),
            block_hash=self._hash_block(0, self.GENESIS_HASH, [], int(time.time()), self.user_address)
        )
        self.blocks.append(genesis_block)
        self._save_chain_state()

    def _hash_block(self, height: int, prev_hash: str, txs: List[Transaction], timestamp: int, validator: str) -> str:
        block_data = f"{height}:{prev_hash}:{[tx.tx_id for tx in txs]}:{timestamp}:{validator}"
        return hashlib.sha256(block_data.encode()).hexdigest()

    def _sign_message(self, message: str) -> str:
        return self.account.sign_message(encode_defunct(text=message)).signature.hex()

    def _verify_signature(self, message: str, signature: str, address: str) -> bool:
        try:
            recovered = Account.recover_message(encode_defunct(text=message), signature=signature)
            return recovered.lower() == address.lower()
        except:
            return False

    def create_transaction(self, type: str, data: Dict, sender: str = None) -> Transaction:
        sender = sender or self.user_address
        timestamp = int(time.time())
        tx_data = {"type": type, "data": data, "sender": sender, "timestamp": timestamp}
        tx_id = hashlib.sha256(json.dumps(tx_data, sort_keys=True).encode()).hexdigest()
        signature = self._sign_message(f"tx:{tx_id}")
        tx = Transaction(tx_id=tx_id, sender=sender, type=type, data=data, timestamp=timestamp, signature=signature)
        self.pending_txs.append(tx)
        self.logger.log("create_tx", json.dumps(tx_data).encode(), tx_id.encode(), {"type": type})
        return tx

    async def _handle_gossip(self, stream):
        try:
            data = await stream.read()
            msg = json.loads(data.decode())
            if msg["type"] == "transaction":
                tx = Transaction(**msg["data"])
                if self._verify_signature(f"tx:{tx.tx_id}", tx.signature, tx.sender):
                    self.pending_txs.append(tx)
            elif msg["type"] == "block":
                block = Block(**msg["data"])
                if self._validate_block(block):
                    self.blocks.append(block)
                    self._apply_block(block)
                    self._save_chain_state()
            elif msg["type"] in ["identity", "voucher", "claim"]:
                if msg["type"] == "identity":
                    identity = msg["data"]
                    if identity["owner"].lower() not in self.soulbound_system.identities:
                        self.soulbound_system.identities[identity["owner"].lower()] = SoulBoundIdentity(**identity)
                        self.soulbound_system._save_identities()
                elif msg["type"] == "voucher":
                    voucher = Voucher(**msg["data"])
                    self.voucher_system.vouchers[voucher.voucher_id] = voucher
                    self.voucher_system._save_vouchers()
                elif msg["type"] == "claim":
                    claim = Claim(**msg["data"])
                    self.consensus.claims[claim.claim_id] = claim
                    self.consensus._save_claims()
        except Exception as e:
            self.logger.log("gossip_error", data, b"", {"error": str(e)}, "ERROR", "GOSSIP_FAILED")
        await stream.close()

    def _validate_block(self, block: Block) -> bool:
        expected_hash = self._hash_block(
            block.block_height, block.previous_hash, block.transactions, block.timestamp, block.validator
        )
        if block.block_hash != expected_hash or block.block_height != len(self.blocks):
            return False
        if block.previous_hash != (self.blocks[-1].block_hash if self.blocks else self.GENESIS_HASH):
            return False
        if not self._verify_signature(f"block:{block.block_height}:{block.block_hash}", block.signature, block.validator):
            return False
        karma = self.yield_protocol.get_karma(block.validator)
        return karma >= self.MIN_KARMA_TO_VALIDATE

    def _apply_block(self, block: Block):
        for tx in block.transactions:
            if tx.type == "create_post":
                post_id = self.town_square.create_post(
                    content=tx.data["content"],
                    author_address=tx.sender,
                    voucher_id=tx.data.get("voucher_id"),
                    signature=tx.signature
                )
                self.chain_state["posts"][post_id] = tx.data
            elif tx.type == "create_pulse":
                pulse_name = self.pulse_system.create_pulse(
                    creator_address=tx.sender,
                    pulse_config=tx.data["config"],
                    signature=tx.signature,
                    voucher_id=tx.data.get("voucher_id")
                )
                self.chain_state["pulses"][pulse_name] = tx.data
            elif tx.type == "submit_validation":
                self.consensus.submit_validation(
                    claim_id=tx.data["claim_id"],
                    validator_address=tx.sender,
                    is_valid=tx.data["is_valid"],
                    comment_id=tx.data.get("comment_id"),
                    signature=tx.signature
                )
                self.chain_state["claims"][tx.data["claim_id"]] = self.consensus.get_consensus_result(tx.data["claim_id"])
            elif tx.type == "mint_identity":
                token_id = self.soulbound_system.mint_identity(
                    owner_address=tx.sender,
                    metadata=tx.data["metadata"],
                    signature=tx.signature,
                    entropy_commitment=bytes.fromhex(tx.data["entropy_commitment"])
                )
                self.chain_state["identities"][tx.sender] = {"token_id": token_id}
            elif tx.type == "verify_identity":
                result = self.soulbound_system.verify_identity(
                    owner_address=tx.sender,
                    signature=tx.signature,
                    zk_did=tx.data["zk_did"],
                    birthdate=tx.data["birthdate"],
                    entropy=bytes.fromhex(tx.data["entropy"]),
                    fee=tx.data["fee"]
                )
                self.chain_state["identities"][tx.sender]["verified"] = result
            elif tx.type == "deposit_erc20":
                voucher_id = self.voucher_system.deposit_erc20(
                    token_address=tx.data["token_address"],
                    amount=tx.data["amount"],
                    password_hash=tx.data["password_hash"],
                    depositor_address=tx.sender,
                    signature=tx.signature
                )
                self.chain_state["vouchers"][voucher_id] = tx.data
            elif tx.type == "redeem_voucher":
                self.voucher_system.redeem_voucher(
                    voucher_id=tx.data["voucher_id"],
                    password=tx.data["password"],
                    recipient_address=tx.data["recipient_address"],
                    redeemer_address=tx.sender,
                    signature=tx.signature
                )
                self.chain_state["vouchers"][tx.data["voucher_id"]]["redeemed"] = True

    async def mine_block(self):
        while True:
            if self.yield_protocol.get_karma(self.user_address) < self.MIN_KARMA_TO_VALIDATE:
                await asyncio.sleep(self.BLOCK_INTERVAL)
                continue
            if not self.pending_txs:
                await asyncio.sleep(self.BLOCK_INTERVAL)
                continue
            block_height = len(self.blocks)
            prev_hash = self.blocks[-1].block_hash if self.blocks else self.GENESIS_HASH
            timestamp = int(time.time())
            txs = self.pending_txs[:10]
            block_hash = self._hash_block(block_height, prev_hash, txs, timestamp, self.user_address)
            signature = self._sign_message(f"block:{block_height}:{block_hash}")
            block = Block(
                block_height=block_height,
                previous_hash=prev_hash,
                transactions=txs,
                timestamp=timestamp,
                validator=self.user_address,
                signature=signature,
                block_hash=block_hash
            )
            if self._validate_block(block):
                self.blocks.append(block)
                self._apply_block(block)
                self.pending_txs = [tx for tx in self.pending_txs if tx not in txs]
                self._save_chain_state()
                await self.p2p.broadcast(json.dumps({"type": "block", "data": vars(block)}).encode())
            await asyncio.sleep(self.BLOCK_INTERVAL)

    async def run_node(self):
        try:
            mining_task = asyncio.create_task(self.mine_block())
            pulse_task = asyncio.create_task(self._run_pulse_loop())
            sync_tasks = [
                asyncio.create_task(self.soulbound_system.sync_identities()),
                asyncio.create_task(self.voucher_system.sync_vouchers()),
                asyncio.create_task(self.consensus.sync_claims())
            ]
            script = """
            const post_content = "Welcome to Onoclast Chain!";
            call create_post(post_content);
            """
            await self.script_interpreter.run(script)
            await asyncio.gather(mining_task, pulse_task, *sync_tasks)
        except Exception as e:
            self.logger.log("node_error", b"", str(e).encode(), {"error": str(e)}, "ERROR", "NODE_RUN_FAILED")

    async def _run_pulse_loop(self):
        global_time = self.pulse_system.state.time
        while True:
            signals = self.pulse_system.process_signals(global_time)
            for sig in signals:
                if sig.name == "create_post":
                    self.create_transaction(
                        type="create_post",
                        data={"content": f"Pulse {sig.from_agent} signal at {time.time()}"}
                    )
            global_time += 0.1
            self.pulse_system.state.time = global_time
            self.pulse_system.state.save_state(f"{self.state_dir}{self.node_id}_pulse_state.json")
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    chain = OnoclastChain(
        node_id="node_1",
        user_address="0xYourUserAddress",
        private_key="0xYourPrivateKey"
    )
    asyncio.run(chain.run_node())