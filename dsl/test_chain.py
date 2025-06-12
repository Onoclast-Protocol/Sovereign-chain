
import json
import time
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .pulse_system import PulseSystem
from ..agents.tcc_logger import TCCLogger

@dataclass
class Transaction:
    tx_id: str
    type: str
    data: Dict
    sender: str
    timestamp: float
    signature: str

@dataclass
class Block:
    block_id: str
    previous_hash: str
    transactions: List[Transaction]
    timestamp: float
    nonce: int
    hash: str

class MockIPFSClient:
    """Mocks IPFS client for testnet storage."""
    def __init__(self):
        self.storage = {}

    def add_bytes(self, data: bytes) -> Dict[str, str]:
        cid = hashlib.sha256(data).hexdigest()
        self.storage[cid] = data
        return {"Hash": cid}

    def cat(self, cid: str) -> bytes:
        return self.storage.get(cid, b"")

class MockSecuritySystem:
    """Mocks BioInspiredSecuritySystem for anomaly detection."""
    def detect_anomaly(self, data: bytes) -> tuple[bool, float, str]:
        # Simplified: reject if 'malicious' in data
        if b"malicious" in data.lower():
            return True, 0.9, "Potential malicious code detected"
        return False, 0.1, "No anomalies detected"

class TestOnoclastChain:
    """Simulates an OnoclastChain testnet for local dApp testing."""
    
    def __init__(
        self,
        node_id: str,
        user_address: str,
        private_key: str,
        log_level: str = "INFO"
    ):
        self.logger = TCCLogger(level=log_level)
        self.node_id = node_id
        self.user_address = user_address
        self.private_key = private_key
        self.ipfs_client = MockIPFSClient()
        self.security = MockSecuritySystem()
        self.pulse_system = PulseSystem(
            node_id=node_id,
            state_dir=":memory:",
            log_level=log_level
        )
        self.blocks: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.state: Dict[str, Any] = {"accounts": {user_address: {"balance": 1000}}}
        self.genesis_block = self._create_genesis_block()
        self.blocks.append(self.genesis_block)

    def _create_genesis_block(self) -> Block:
        """Creates the genesis block for the test chain."""
        block_id = "genesis"
        timestamp = time.time()
        data = json.dumps({"block_id": block_id, "timestamp": timestamp}).encode()
        block_hash = hashlib.sha256(data).hexdigest()
        return Block(
            block_id=block_id,
            previous_hash="0" * 64,
            transactions=[],
            timestamp=timestamp,
            nonce=0,
            hash=block_hash
        )

    def create_transaction(self, type: str, data: Dict, sender: str) -> Transaction:
        """Creates a transaction for the test chain."""
        try:
            tx_id = hashlib.sha256(
                json.dumps({"type": type, "data": data, "sender": sender, "time": time.time()}).encode()
            ).hexdigest()
            signature = self._sign_message(f"tx:{tx_id}", self.private_key)
            tx = Transaction(
                tx_id=tx_id,
                type=type,
                data=data,
                sender=sender,
                timestamp=time.time(),
                signature=signature
            )
            self.pending_transactions.append(tx)
            self.logger.log("tx_created", tx_id.encode(), json.dumps(data).encode(),
                          {"type": type, "sender": sender})
            return tx
        except Exception as e:
            self.logger.log("tx_create_error", b"", str(e).encode(),
                          {"error": str(e)}, "ERROR", "TX_CREATE")
            raise

    def validate_transaction(self, tx: Transaction) -> bool:
        """Validates a transaction for the test chain."""
        try:
            if tx.sender not in self.state["accounts"]:
                return False
            if tx.type == "deploy_script":
                if not tx.data.get("wasm_id") or not tx.data.get("script"):
                    return False
            expected_sig = self._sign_message(f"tx:{tx.tx_id}", self.private_key)
            if tx.signature != expected_sig:
                return False
            return True
        except Exception as e:
            self.logger.log("tx_validate_error", tx.tx_id.encode(), str(e).encode(),
                          {"error": str(e)}, "ERROR", "TX_VALIDATE")
            return False

    def mine_block(self) -> Block:
        """Mines a new block with pending transactions."""
        try:
            previous_block = self.blocks[-1]
            transactions = self.pending_transactions.copy()
            self.pending_transactions = []
            
            block_id = hashlib.sha256(
                json.dumps({"previous_hash": previous_block.hash, "time": time.time()}).encode()
            ).hexdigest()
            timestamp = time.time()
            nonce = 0
            block_data = {
                "block_id": block_id,
                "previous_hash": previous_block.hash,
                "transactions": [vars(tx) for tx in transactions],
                "timestamp": timestamp,
                "nonce": nonce
            }
            block_hash = hashlib.sha256(json.dumps(block_data).encode()).hexdigest()
            
            block = Block(
                block_id=block_id,
                previous_hash=previous_block.hash,
                transactions=transactions,
                timestamp=timestamp,
                nonce=nonce,
                hash=block_hash
            )
            self.blocks.append(block)
            
            # Update state for deploy_script transactions
            for tx in transactions:
                if tx.type == "deploy_script":
                    self.state.setdefault("deployed_scripts", {})[tx.tx_id] = tx.data
            
            self.logger.log("block_mined", block_id.encode(), block_hash.encode(),
                          {"tx_count": len(transactions)})
            return block
        except Exception as e:
            self.logger.log("block_mine_error", b"", str(e).encode(),
                          {"error": str(e)}, "ERROR", "BLOCK_MINE")
            raise

    def _sign_message(self, message: str, private_key: str) -> str:
        """Mocks message signing for the test chain."""
        # Simplified: hash message with private_key as salt
        return hashlib.sha256((message + private_key).encode()).hexdigest()

    async def run_node(self):
        """Simulates running the test chain node."""
        try:
            self.logger.log("node_start", self.node_id.encode(), b"Test chain started",
                          {"node_id": self.node_id})
            while True:
                if self.pending_transactions:
                    self.mine_block()
                await asyncio.sleep(1.0)  # Simulate block mining interval
        except Exception as e:
            self.logger.log("node_run_error", self.node_id.encode(), str(e).encode(),
                          {"error": str(e)}, "ERROR", "NODE_RUN")
            raise

    def get_state(self) -> Dict[str, Any]:
        """Returns the current chain state for debugging."""
        return {
            "blocks": [vars(block) for block in self.blocks],
            "pending_transactions": [vars(tx) for tx in self.pending_transactions],
            "state": self.state
        }

if __name__ == "__main__":
    # Example usage
    test_chain = TestOnoclastChain(
        node_id="test_node",
        user_address="0xTestAddress",
        private_key="0xTestPrivateKey"
    )
    tx = test_chain.create_transaction(
        type="deploy_script",
        data={"wasm_id": "test_wasm_cid", "script": "let x = 42; print x;"},
        sender="0xTestAddress"
    )
    block = test_chain.mine_block()
    print(json.dumps(test_chain.get_state(), indent=2))
