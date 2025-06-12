import json
import re
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from web3 import Web3
from .town_square import TownSquare
from .soulbound_identity import SoulBoundIdentitySystem
from .yield_protocol import YieldProtocol
from ..agents.tcc_logger import TCCLogger

# Load DSL specification
with open("dsl/grammar.json", "r") as f:
    DSL_SPEC = json.load(f)

@dataclass
class Variable:
    name: str
    value: Any
    is_const: bool = False

@dataclass
class Function:
    name: str
    args: List[str]
    body: List[Dict]

class SovereignScriptInterpreter:
    def __init__(
        self,
        town_square: TownSquare,
        soulbound_system: SoulBoundIdentitySystem,
        yield_protocol: YieldProtocol,
        user_address: str,
        private_key: str,
        log_level: str = "INFO"
    ):
        self.logger = TCCLogger(level=log_level)
        self.town_square = town_square
        self.soulbound_system = soulbound_system
        self.yield_protocol = yield_protocol
        self.user_address = user_address
        self.w3 = Web3(Web3.HTTPProvider("https://arb1.arbitrum.io/rpc"))
        self.account = self.w3.eth.account.from_key(private_key)
        self.variables: Dict[str, Variable] = {}
        self.functions: Dict[str, Function] = {}
        self.components: Dict[str, Dict] = {}
        self.state: Dict[str, Any] = {}
        self.event_handlers: Dict[str, Callable] = {}

    def parse(self, code: str) -> List[Dict]:
        """Parse SovereignScript code into an AST."""
        lines = code.strip().split("\n")
        ast = []
        current_block = None
        block_stack = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match variable declarations and updates
            for decl in DSL_SPEC["structure"]["variable"]["declare"]:
                pattern = decl.replace("<name>", r"(\w+)").replace("<value>", r"(.+?)").replace(";", r"\s*;")
                if re.match(pattern, line):
                    name, value = re.match(pattern, line).groups()
                    ast.append({
                        "type": "variable_decl",
                        "name": name,
                        "value": value,
                        "is_const": "const" in decl
                    })
                    continue

            for update in DSL_SPEC["structure"]["variable"]["update"]:
                pattern = update.replace("<name>", r"(\w+)").replace("<value>", r"(.+?)").replace(";", r"\s*;")
                if re.match(pattern, line):
                    name, value = re.match(pattern, line).groups()
                    ast.append({
                        "type": "variable_update",
                        "name": name,
                        "value": value
                    })
                    continue

            # Match output statements
            for out in DSL_SPEC["output"]:
                pattern = out.replace("<value>", r"(.+?)").replace(";", r"\s*;")
                if re.match(pattern, line):
                    value = re.match(pattern, line).groups()[0]
                    ast.append({
                        "type": "output",
                        "value": value
                    })
                    continue

            # Match function definitions
            for func_def in DSL_SPEC["function"]["define"]:
                pattern = func_def.replace("<name>", r"(\w+)").replace("<args>", r"\((.*?)\)")
                if re.match(pattern, line):
                    name, args = re.match(pattern, line).groups()
                    args = [arg.strip() for arg in args.split(",") if arg.strip()]
                    current_block = {
                        "type": "function_def",
                        "name": name,
                        "args": args,
                        "body": []
                    }
                    block_stack.append(current_block)
                    continue

            # Match function calls
            for call in DSL_SPEC["function"]["call"]:
                pattern = call.replace("<name>", r"(\w+)").replace("<args>", r"\((.*?)\)").replace(";", r"\s*;")
                if re.match(pattern, line):
                    name, args = re.match(pattern, line).groups()
                    args = [arg.strip() for arg in args.split(",") if arg.strip()]
                    ast.append({
                        "type": "function_call",
                        "name": name,
                        "args": args
                    })
                    continue

            # Match control flow (if, loop)
            for if_stmt in DSL_SPEC["control"]["if"]:
                pattern = if_stmt.replace("<condition>", r"(.+?)")
                if re.match(pattern, line):
                    condition = re.match(pattern, line).groups()[0]
                    current_block = {
                        "type": "if",
                        "condition": condition,
                        "body": []
                    }
                    block_stack.append(current_block)
                    continue

            for loop in DSL_SPEC["control"]["loop"]:
                pattern = loop.replace("<condition>", r"(.+?)")
                if re.match(pattern, line):
                    condition = re.match(pattern, line).groups()[0]
                    current_block = {
                        "type": "loop",
                        "condition": condition,
                        "body": []
                    }
                    block_stack.append(current_block)
                    continue

            # Match UI components
            for comp in DSL_SPEC["ui"]["component"]:
                pattern = comp.replace("<name>", r"(\w+)").replace("{ <body> }", r"\{(.*)\}")
                if re.match(pattern, line, re.DOTALL):
                    name, body = re.match(pattern, line, re.DOTALL).groups()
                    current_block = {
                        "type": "component",
                        "name": name,
                        "body": self.parse(body)
                    }
                    block_stack.append(current_block)
                    continue

            # Match block endings
            if line in DSL_SPEC["control"]["end"] + DSL_SPEC["function"]["end"]:
                if block_stack:
                    block = block_stack.pop()
                    if block_stack:
                        block_stack[-1]["body"].append(block)
                    else:
                        ast.append(block)
                    current_block = block_stack[-1] if block_stack else None
                continue

        return ast

    def evaluate_expression(self, expr: str) -> Any:
        """Evaluate a simple expression (supports variables and basic math)."""
        if expr in self.variables:
            return self.variables[expr].value
        try:
            return eval(expr, {"__builtins__": {}}, {k: v.value for k, v in self.variables.items()})
        except:
            return expr

    def execute(self, ast: List[Dict], context: Dict[str, Any] = None) -> Any:
        """Execute the AST in the current context."""
        context = context or {}
        result = None

        for node in ast:
            start_time = time.time_ns()

            if node["type"] == "variable_decl":
                value = self.evaluate_expression(node["value"])
                self.variables[node["name"]] = Variable(
                    name=node["name"],
                    value=value,
                    is_const=node.get("is_const", False)
                )
                self.logger.log(
                    "variable_decl",
                    node["name"].encode(),
                    str(value).encode(),
                    {"execution_time_ns": time.time_ns() - start_time}
                )

            elif node["type"] == "variable_update":
                if node["name"] not in self.variables:
                    raise ValueError(f"Variable {node['name']} not found")
                if self.variables[node["name"]].is_const:
                    raise ValueError(f"Cannot update constant {node['name']}")
                value = self.evaluate_expression(node["value"])
                self.variables[node["name"]].value = value
                self.logger.log(
                    "variable_update",
                    node["name"].encode(),
                    str(value).encode(),
                    {"execution_time_ns": time.time_ns() - start_time}
                )

            elif node["type"] == "output":
                value = self.evaluate_expression(node["value"])
                print(value)
                self.logger.log(
                    "output",
                    node["value"].encode(),
                    str(value).encode(),
                    {"execution_time_ns": time.time_ns() - start_time}
                )

            elif node["type"] == "function_def":
                self.functions[node["name"]] = Function(
                    name=node["name"],
                    args=node["args"],
                    body=node["body"]
                )
                self.logger.log(
                    "function_def",
                    node["name"].encode(),
                    b"",
                    {"args": node["args"], "execution_time_ns": time.time_ns() - start_time}
                )

            elif node["type"] == "function_call":
                if node["name"] not in self.functions:
                    raise ValueError(f"Function {node['name']} not found")
                func = self.functions[node["name"]]
                if len(node["args"]) != len(func.args):
                    raise ValueError(f"Argument mismatch for {node['name']}")
                local_context = {k: self.evaluate_expression(v) for k, v in zip(func.args, node["args"])}
                result = self.execute(func.body, local_context)
                self.logger.log(
                    "function_call",
                    node["name"].encode(),
                    b"",
                    {"args": node["args"], "execution_time_ns": time.time_ns() - start_time}
                )

            elif node["type"] == "if":
                condition = self.evaluate_expression(node["condition"])
                if condition:
                    result = self.execute(node["body"], context)
                self.logger.log(
                    "if",
                    node["condition"].encode(),
                    b"",
                    {"condition_result": bool(condition), "execution_time_ns": time.time_ns() - start_time}
                )

            elif node["type"] == "loop":
                while self.evaluate_expression(node["condition"]):
                    result = self.execute(node["body"], context)
                self.logger.log(
                    "loop",
                    node["condition"].encode(),
                    b"",
                    {"execution_time_ns": time.time_ns() - start_time}
                )

            elif node["type"] == "component":
                self.components[node["name"]] = {
                    "body": node["body"],
                    "state": {},
                    "props": {}
                }
                self.logger.log(
                    "component",
                    node["name"].encode(),
                    b"",
                    {"execution_time_ns": time.time_ns() - start_time}
                )

            # Blockchain integration: Create a post
            elif node["type"] == "function_call" and node["name"] == "create_post":
                content = self.evaluate_expression(node["args"][0])
                signature = self.w3.eth.account.sign_message(
                    encode_defunct(text=f"create_post:{content}:0"),
                    private_key=self.account.key
                ).signature.hex()
                post_id = self.town_square.create_post(
                    content=content,
                    author_address=self.user_address,
                    voucher_id=None,
                    signature=signature
                )
                result = post_id
                self.logger.log(
                    "create_post",
                    content.encode(),
                    str(post_id).encode(),
                    {"execution_time_ns": time.time_ns() - start_time}
                )

        return result

    def run(self, code: str) -> Any:
        """Run SovereignScript code."""
        try:
            ast = self.parse(code)
            return self.execute(ast)
        except Exception as e:
            self.logger.log(
                "run_error",
                code.encode(),
                str(e).encode(),
                {"error": str(e)},
                "ERROR",
                "RUN_FAILED"
            )
            raise

# Example usage
if __name__ == "__main__":
    # Mock dependencies
    town_square = TownSquare(...)  # Initialize with actual dependencies
    soulbound_system = SoulBoundIdentitySystem(...)
    yield_protocol = YieldProtocol(...)
    interpreter = SovereignScriptInterpreter(
        town_square=town_square,
        soulbound_system=soulbound_system,
        yield_protocol=yield_protocol,
        user_address="0xYourUserAddress",
        private_key="0xYourPrivateKey"
    )

    # Example SovereignScript code
    code = """
    const greeting = "Hello, Onoclast!";
    say greeting;
    fn create_greeting_post(content) {
        call create_post(content);
    }
    call create_greeting_post(greeting);
    """
    interpreter.run(code)