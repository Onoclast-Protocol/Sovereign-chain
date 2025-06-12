
import os
import json
import hashlib
from typing import Dict, List, Any, Optional
from wasmer import Store, Module, Instance
from wasmer_compiler_cranelift import Compiler
import ipfshttpclient
from .parser import DSLParser
from .onoclast_chain import OnoclastChain
from ..agents.tcc_logger import TCCLogger

class WASMCompiler:
    """Compiles SovereignScript-v3.4 AST to WASM bytecode for Onoclast Protocol."""
    
    def __init__(
        self,
        dsl_spec: Dict,
        ipfs_endpoint: str = "/ip4/127.0.0.1/tcp/5001",
        chain: Optional[OnoclastChain] = None,
        log_level: str = "INFO"
    ):
        self.logger = TCCLogger(level=log_level)
        self.dsl_spec = dsl_spec
        self.parser = DSLParser(dsl_spec)
        self.chain = chain
        self.store = Store(Compiler)
        
        try:
            self.ipfs_client = ipfshttpclient.connect(ipfs_endpoint)
        except Exception as e:
            self.logger.log("ipfs_init", b"", str(e).encode(), {"error": str(e)}, "ERROR", "IPFS_INIT")
            raise RuntimeError(f"Failed to connect to IPFS: {e}")

        # WASM constants
        self.WASM_MAGIC = b"\x00\x61\x73\x6d"  # WASM magic number
        self.WASM_VERSION = b"\x01\x00\x00\x00"  # Version 1
        self.type_section = []
        self.function_section = []
        self.code_section = []
        self.export_section = []
        self.local_count = 0
        self.func_index = 0

    def compile(self, code: str) -> Dict[str, Any]:
        """Compiles SovereignScript code to WASM, storing in IPFS."""
        try:
            # Security check
            if self.chain:
                is_anomaly, score, explanation = self.chain.security.detect_anomaly(code.encode())
                if is_anomaly:
                    self.logger.log("compile_anomaly", code.encode(), explanation.encode(), 
                                  {"score": score}, "ERROR", "ANOMALY")
                    return {"wasm_id": None, "error": f"Anomaly detected: {explanation}"}

            # Parse code to AST
            ast = self.parser.parse(code)
            self.logger.log("parse_ast", code.encode(), json.dumps(ast).encode(), 
                          {"action": "parse"})

            # Reset compiler state
            self.type_section = []
            self.function_section = []
            self.code_section = []
            self.export_section = []
            self.local_count = 0
            self.func_index = 0

            # Generate WASM
            wasm_code = self._generate_wasm(ast)
            
            # Validate WASM
            module = Module(self.store, wasm_code)
            instance = Instance(module)  # Test instantiation
            wasm_hash = hashlib.sha256(wasm_code).hexdigest()

            # Store in IPFS
            ipfs_result = self.ipfs_client.add_bytes(wasm_code)
            wasm_id = ipfs_result["Hash"]

            self.logger.log("compile_success", code.encode(), wasm_id.encode(), 
                          {"wasm_hash": wasm_hash, "ipfs_cid": wasm_id})
            return {"wasm_id": wasm_id, "wasm_code": wasm_code.hex(), "error": None}
        
        except Exception as e:
            self.logger.log("compile_error", code.encode(), str(e).encode(), 
                          {"error": str(e)}, "ERROR", "COMPILE")
            return {"wasm_id": None, "wasm_code": None, "error": str(e)}

    def _generate_wasm(self, ast: Dict) -> bytes:
        """Generates WASM bytecode from SovereignScript AST."""
        # Initialize sections
        self._add_main_function(ast)

        # Build WASM module
        sections = [
            self.WASM_MAGIC + self.WASM_VERSION,
            self._encode_type_section(),
            self._encode_function_section(),
            self._encode_export_section(),
            self._encode_code_section()
        ]
        wasm_code = b"".join(sections)
        return wasm_code

    def _add_main_function(self, ast: Dict):
        """Generates the main function from AST nodes."""
        func_type = [0x60, 0x00, 0x00]  # (func () -> ())
        self.type_section.append(func_type)
        
        self.function_section.append(len(self.type_section) - 1)  # Type index
        body = []
        
        for node in ast.get("nodes", []):
            body.extend(self._compile_node(node))
        
        body.append(0x0b)  # end
        code = [self._encode_varuint(self.local_count)] + body
        self.code_section.append(code)
        
        # Export main function
        export_name = b"main"
        self.export_section.append([len(export_name)] + list(export_name) + [0x00, self.func_index])
        self.func_index += 1

    def _compile_node(self, node: Dict) -> List[int]:
        """Compiles an AST node to WASM opcodes."""
        node_type = node.get("type")
        bytecode = []

        if node_type == "variable_declare":
            self.local_count += 1
            local_idx = self.local_count - 1
            value = node.get("value")
            bytecode.extend(self._compile_expression(value))
            bytecode.append(0x21)  # local.set
            bytecode.append(local_idx)

        elif node_type == "output":
            value = node.get("value")
            bytecode.extend(self._compile_expression(value))
            bytecode.append(0x41)  # i32.const (placeholder for print)
            bytecode.append(0x00)  # Dummy value
            bytecode.append(0x10)  # call
            bytecode.append(0x00)  # Function index (assumes print function)

        elif node_type == "function_define":
            self._add_function(node)

        elif node_type == "function_call":
            func_name = node.get("name")
            func_idx = self._get_function_index(func_name)
            bytecode.append(0x10)  # call
            bytecode.append(func_idx)

        elif node_type == "control_if":
            condition = node.get("condition")
            then_block = node.get("then")
            else_block = node.get("else")
            
            bytecode.extend(self._compile_expression(condition))
            bytecode.append(0x04)  # if
            bytecode.append(0x40)  # blocktype: void
            for stmt in then_block.get("nodes", []):
                bytecode.extend(self._compile_node(stmt))
            if else_block:
                bytecode.append(0x05)  # else
                for stmt in else_block.get("nodes", []):
                    bytecode.extend(self._compile_node(stmt))
            bytecode.append(0x0b)  # end

        elif node_type == "control_loop":
            condition = node.get("condition")
            body = node.get("body")
            
            bytecode.append(0x03)  # loop
            bytecode.append(0x40)  # blocktype: void
            bytecode.extend(self._compile_expression(condition))
            bytecode.append(0x0c)  # br_if
            bytecode.append(0x01)  # Break to loop start
            for stmt in body.get("nodes", []):
                bytecode.extend(self._compile_node(stmt))
            bytecode.append(0x0c)  # br
            bytecode.append(0x00)  # Loop back
            bytecode.append(0x0b)  # end

        elif node_type == "fractal_define":
            # Placeholder: Store fractal metadata for runtime rendering
            bytecode.append(0x41)  # i32.const
            bytecode.append(0x00)  # Dummy fractal ID
            bytecode.append(0x10)  # call
            bytecode.append(0x01)  # Assume fractal render function

        return bytecode

    def _compile_expression(self, expr: Any) -> List[int]:
        """Compiles an expression to WASM opcodes."""
        if isinstance(expr, dict) and expr.get("type") == "literal":
            value = expr.get("value")
            if isinstance(value, int):
                return [0x41, self._encode_varint(value)]  # i32.const
            elif isinstance(value, float):
                return [0x43, *self._encode_f32(value)]  # f32.const
            elif isinstance(value, str):
                return [0x41, 0x00]  # Placeholder for string (i32.const 0)
        elif isinstance(expr, dict) and expr.get("type") == "binary_op":
            left = self._compile_expression(expr.get("left"))
            right = self._compile_expression(expr.get("right"))
            op = expr.get("operator")
            op_codes = {
                "+": 0x6a,  # i32.add
                "-": 0x6b,  # i32.sub
                "*": 0x6c,  # i32.mul
                "/": 0x6d   # i32.div_s
            }
            return left + right + [op_codes.get(op, 0x6a)]
        return [0x41, 0x00]  # Default: i32.const 0

    def _add_function(self, node: Dict):
        """Generates a WASM function from a function definition."""
        func_name = node.get("name")
        params = node.get("params", [])
        body = node.get("body", {"nodes": []})
        
        # Function type: (params -> void)
        param_count = len(params)
        func_type = [0x60, param_count, *[0x7f] * param_count, 0x00]  # i32 params, void return
        self.type_section.append(func_type)
        
        # Function declaration
        self.function_section.append(len(self.type_section) - 1)
        
        # Function body
        local_start = self.local_count
        self.local_count += param_count
        code = []
        for stmt in body.get("nodes", []):
            code.extend(self._compile_node(stmt))
        code.append(0x0b)  # end
        
        # Encode locals
        locals = [self._encode_varuint(self.local_count - local_start)]
        self.code_section.append(locals + code)
        
        # Export function
        export_name = func_name.encode()
        self.export_section.append([len(export_name)] + list(export_name) + [0x00, self.func_index])
        self.func_index += 1

    def _get_function_index(self, func_name: str) -> int:
        """Returns the index of a function by name."""
        for export in self.export_section:
            name_len = export[0]
            name = bytes(export[1:1+name_len]).decode()
            if name == func_name:
                return export[-1]
        return 0  # Default to index 0 (main)

    def _encode_type_section(self) -> bytes:
        """Encodes the WASM type section."""
        if not self.type_section:
            return b""
        section = [0x01]  # Type section ID
        types = [len(self.type_section)] + [b for t in self.type_section for b in t]
        section.extend(self._encode_varuint(len(types)))
        section.extend(types)
        return bytes(section)

    def _encode_function_section(self) -> bytes:
        """Encodes the WASM function section."""
        if not self.function_section:
            return b""
        section = [0x03]  # Function section ID
        section.extend(self._encode_varuint(len(self.function_section)))
        section.extend(self._encode_varuint(i) for i in self.function_section)
        return bytes(section)

    def _encode_export_section(self) -> bytes:
        """Encodes the WASM export section."""
        if not self.export_section:
            return b""
        section = [0x07]  # Export section ID
        exports = [len(self.export_section)] + [b for e in self.export_section for b in e]
        section.extend(self._encode_varuint(len(exports)))
        section.extend(exports)
        return bytes(section)

    def _encode_code_section(self) -> bytes:
        """Encodes the WASM code section."""
        section = [0x0a]  # Code section ID
        codes = [len(self.code_section)] + [b for c in self.code_section for b in self._encode_code(c)]
        section.extend(self._encode_varuint(len(codes)))
        section.extend(codes)
        return bytes(section)

    def _encode_code(self, code: List[int]) -> List[int]:
        """Encodes a single function’s code."""
        return [self._encode_varuint(len(code))] + code

    def _encode_varuint(self, n: int) -> List[int]:
        """Encodes an unsigned integer in LEB128 format."""
        bytes_ = []
        while True:
            byte = n & 0x7f
            n >>= 7
            if n == 0:
                bytes_.append(byte)
                break
            bytes_.append(byte | 0x80)
        return bytes_

    def _encode_varint(self, n: int) -> int:
        """Encodes a signed integer in LEB128 format (simplified)."""
        return self._encode_varuint(n if n >= 0 else (1 << 32) + n)[0]

    def _encode_f32(self, f: float) -> List[int]:
        """Encodes a 32-bit float in little-endian format."""
        from struct import pack
        return list(pack("<f", f))

if __name__ == "__main__":
    # Example usage
    dsl_spec = {
        "language": "SovereignScript-v3.4",
        "symbols": {"assign": "=", "math": ["+", "-", "*", "/"], "end": [";"]},
        "structure": {
            "variable": {"declare": ["let <name> = <value>;"]},
            "output": ["print <value>;"],
            "function": {"define": ["fn <name>(<args>)"], "call": ["call <name>(<args>);"], "end": ["end"]},
            "control": {"if": ["if <condition>"], "loop": ["loop while <condition>"], "end": ["end"]}
        }
    }
    compiler = WASMCompiler(dsl_spec)
    code = """
    let x = 42;
    print x;
    fn add(a, b)
        let sum = a + b;
        print sum;
    end
    call add(10, 20);
    """
    result = compiler.compile(code)
    print(json.dumps(result, indent=2))
