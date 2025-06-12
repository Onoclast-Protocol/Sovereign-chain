
import json
import hashlib
from typing import Dict, Any, Optional
from .onoclast_chain import OnoclastChain
from .pulse_system import PulseSystem
from .compiler import WASMCompiler
from ..agents.tcc_logger import TCCLogger

class SovereignScriptDeployer:
    """Deploys SovereignScript-v3.4 WASM to OnoclastChain for the IDE."""
    
    def __init__(
        self,
        compiler: WASMCompiler,
        chain: OnoclastChain,
        pulse_system: PulseSystem,
        user_address: str,
        private_key: str,
        log_level: str = "INFO"
    ):
        self.logger = TCCLogger(level=log_level)
        self.compiler = compiler
        self.chain = chain
        self.pulse_system = pulse_system
        self.user_address = user_address
        self.private_key = private_key

    def deploy(self, code: str) -> Dict[str, Any]:
        """Deploys SovereignScript code as WASM to OnoclastChain."""
        try:
            # Compile code to WASM
            compile_result = self.compiler.compile(code)
            if compile_result["error"]:
                self.logger.log("deploy_compile_error", code.encode(), compile_result["error"].encode(),
                              {"error": compile_result["error"]}, "ERROR", "DEPLOY_COMPILE")
                return {"tx_id": None, "error": compile_result["error"]}

            wasm_id = compile_result["wasm_id"]
            wasm_code = bytes.fromhex(compile_result["wasm_code"])

            # Security check (already done in compiler, but recheck for safety)
            is_anomaly, score, explanation = self.chain.security.detect_anomaly(code.encode())
            if is_anomaly:
                self.logger.log("deploy_anomaly", code.encode(), explanation.encode(),
                              {"score": score}, "ERROR", "ANOMALY")
                return {"tx_id": None, "error": f"Anomaly detected: {explanation}"}

            # Prepare transaction
            tx_data = {
                "wasm_id": wasm_id,
                "script": code,
                "metadata": {
                    "language": "SovereignScript-v3.4",
                    "version": "1.0",
                    "wasm_hash": hashlib.sha256(wasm_code).hexdigest()
                }
            }

            # Create and validate transaction
            tx = self.chain.create_transaction(
                type="deploy_script",
                data=tx_data,
                sender=self.user_address
            )
            if not self._validate_transaction(tx):
                self.logger.log("deploy_validation_error", code.encode(), b"Invalid transaction",
                              {"tx_id": tx.tx_id}, "ERROR", "DEPLOY_VALIDATE")
                return {"tx_id": None, "error": "Transaction validation failed"}

            # Store transaction metadata in IPFS
            ipfs_result = self.chain.ipfs_client.add_bytes(json.dumps(tx_data).encode())
            tx_data["ipfs_metadata"] = ipfs_result["Hash"]

            # Emit PulseSystem signal
            pulse_config = {
                "name": f"deploy_{tx.tx_id}",
                "interval": 1.0,
                "next_fire": self.pulse_system.state.time + 1.0,
                "body": [
                    {"type": "emit", "signal": "create_post", "to": "town_square", "strength": 1.0,
                     "data": {"tx_id": tx.tx_id, "wasm_id": wasm_id}}
                ],
                "time_scale": 1.0,
                "fractions": 1,
                "enabled": True
            }
            signature = self.chain._sign_message(
                f"create_pulse:{json.dumps(pulse_config, sort_keys=True)}:0",
                self.private_key
            )
            self.pulse_system.create_pulse(
                creator_address=self.user_address,
                pulse_config=pulse_config,
                signature=signature
            )

            self.logger.log("deploy_success", code.encode(), tx.tx_id.encode(),
                          {"wasm_id": wasm_id, "tx_id": tx.tx_id, "ipfs_metadata": tx_data["ipfs_metadata"]})
            return {"tx_id": tx.tx_id, "error": None}

        except Exception as e:
            self.logger.log("deploy_error", code.encode(), str(e).encode(),
                          {"error": str(e)}, "ERROR", "DEPLOY")
            return {"tx_id": None, "error": str(e)}

    def _validate_transaction(self, tx: Any) -> bool:
        """Validates a transaction using OnoclastChain consensus."""
        try:
            # Simplified validation: check sender, data integrity, and chain consensus
            if tx.sender != self.user_address:
                return False
            if not self.chain.validate_transaction(tx):
                return False
            # Additional checks (e.g., gas limits, signature) can be added here
            return True
        except Exception as e:
            self.logger.log("validate_tx_error", tx.tx_id.encode(), str(e).encode(),
                          {"error": str(e)}, "ERROR", "VALIDATE_TX")
            return False

if __name__ == "__main__":
    # Example usage
    from .compiler import WASMCompiler
    dsl_spec = {
        "language": "SovereignScript-v3.4",
        "symbols": {"assign": "=", "math": ["+", "-", "*", "/"], "end": [";"]},
        "structure": {
            "variable": {"declare": ["let <name> = <value>;"]},
            "output": ["print <value>;"],
            "function": {"define": ["fn <name>(<args>)"], "call": ["call <name>(<args>);"], "end": ["end"]}
        }
    }
    chain = OnoclastChain(
        node_id="node_1",
        user_address="0xTestAddress",
        private_key="0xTestPrivateKey",
        ipfs_endpoint="/ip4/127.0.0.1/tcp/5001",
        state_dir="data/chain_state/",
        log_level="INFO"
    )
    compiler = WASMCompiler(dsl_spec, ipfs_endpoint="/ip4/127.0.0.1/tcp/5001", chain=chain)
    deployer = SovereignScriptDeployer(
        compiler=compiler,
        chain=chain,
        pulse_system=chain.pulse_system,
        user_address="0xTestAddress",
        private_key="0xTestPrivateKey"
    )
    code = """
    let x = 42;
    print x;
    fn add(a, b)
        let sum = a + b;
        print sum;
    end
    """
    result = deployer.deploy(code)
    print(json.dumps(result, indent=2))
