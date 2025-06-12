
import json
from typing import Dict, Any, Optional
from .parser import DSLParser
from .swarm_dsl_runtime import SwarmDSLRuntime
from .onoclast_chain import OnoclastChain
from .pulse_system import PulseSystem
from ..agents.tcc_logger import TCCLogger

class SovereignScriptInterpreter:
    """Interprets SovereignScript-v3.4 code for local execution in the Onoclast IDE."""
    
    def __init__(
        self,
        dsl_spec: Dict,
        chain: Optional[OnoclastChain] = None,
        pulse_system: Optional[PulseSystem] = None,
        log_level: str = "INFO"
    ):
        self.logger = TCCLogger(level=log_level)
        self.dsl_spec = dsl_spec
        self.parser = DSLParser(dsl_spec)
        self.runtime = SwarmDSLRuntime(dsl_spec)
        self.chain = chain
        self.pulse_system = pulse_system
        self.state: Dict[str, Any] = {"variables": {}, "functions": {}, "ui_state": {}, "fractals": {}}

    def execute(self, code: str, pulse_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Executes SovereignScript code locally, returning output or errors."""
        try:
            # Security validation
            if self.chain:
                is_anomaly, score, explanation = self.chain.security.detect_anomaly(code.encode())
                if is_anomaly:
                    self.logger.log("execute_anomaly", code.encode(), explanation.encode(),
                                  {"score": score}, "ERROR", "ANOMALY")
                    return {"output": None, "error": f"Anomaly detected: {explanation}"}

            # Parse code to AST
            ast = self.parser.parse(code)
            self.logger.log("execute_parse", code.encode(), json.dumps(ast).encode(),
                          {"action": "execute_parse"})

            # Initialize output
            output: List[str] = []

            # Process AST nodes
            for node in ast.get("nodes", []):
                result = self._process_node(node, output)
                if result.get("error"):
                    self.logger.log("execute_node_error", code.encode(), result["error"].encode(),
                                  {"node_type": node.get("type")}, "ERROR", "EXECUTE_NODE")
                    return {"output": None, "error": result["error"]}

            # Apply pulse data for dynamic updates
            if pulse_data and self.pulse_system:
                self._apply_pulse_data(pulse_data, output)

            # Combine output
            output_str = "\n".join(output) if output else "No output"
            self.logger.log("execute_success", code.encode(), output_str.encode(),
                          {"state": self.state})
            return {"output": output_str, "error": None}

        except Exception as e:
            self.logger.log("execute_error", code.encode(), str(e).encode(),
                          {"error": str(e)}, "ERROR", "EXECUTE")
            return {"output": None, "error": str(e)}

    def _process_node(self, node: Dict, output: List[str]) -> Dict[str, Any]:
        """Processes an AST node, updating state and output."""
        node_type = node.get("type")
        
        if node_type == "variable":
            name = node.get("name")
            value = self._evaluate_expression(node.get("value"))
            self.state["variables"][name] = value
            return {"error": None}

        elif node_type == "output":
            value = self._evaluate_expression(node.get("value"))
            output.append(str(value))
            return {"error": None}

        elif node_type == "function_define":
            name = node.get("name")
            args = node.get("args", [])
            body = node.get("body", {"nodes": []})
            self.state["functions"][name] = {"args": args, "body": body}
            return {"error": None}

        elif node_type == "function_call":
            name = node.get("name")
            args = [self._evaluate_expression(arg) for arg in node.get("args", [])]
            func = self.state["functions"].get(name)
            if not func:
                return {"error": f"Function {name} not defined"}
            return self._execute_function(func, args, output)

        elif node_type == "control_if":
            condition = self._evaluate_expression(node.get("condition"))
            if condition:
                return self._execute_body(node.get("body", {"nodes": []}), output)
            elif node.get("else"):
                return self._execute_body(node["else"], output)
            return {"error": None}

        elif node_type == "control_loop":
            condition = self._evaluate_expression(node.get("condition"))
            while condition:
                result = self._execute_body(node.get("body", {"nodes": []}), output)
                if result.get("error"):
                    return result
                condition = self._evaluate_expression(node.get("condition"))
            return {"error": None}

        elif node_type == "ui_component":
            name = node.get("name")
            state = node.get("state", {})
            self.state["ui_state"][name] = state
            return {"error": None}

        elif node_type == "fractal_define":
            name = node.get("name")
            params = node.get("params", {})
            self.state["fractals"][name] = params
            return {"error": None}

        return {"error": f"Unsupported node type: {node_type}"}

    def _evaluate_expression(self, expr: Any) -> Any:
        """Evaluates an expression, resolving variables and math operations."""
        if isinstance(expr, dict):
            if expr.get("type") == "identifier":
                return self.state["variables"].get(expr["name"], None)
            elif expr.get("type") == "math":
                left = self._evaluate_expression(expr.get("left"))
                right = self._evaluate_expression(expr.get("right"))
                op = expr.get("op")
                if op == "+": return left + right
                if op == "-": return left - right
                if op == "*": return left * right
                if op == "/": return left / right if right != 0 else {"error": "Division by zero"}
            elif expr.get("type") == "literal":
                return expr.get("value")
        return expr

    def _execute_function(self, func: Dict, args: List, output: List[str]) -> Dict[str, Any]:
        """Executes a function with given arguments."""
        local_state = self.state.copy()
        for param, arg in zip(func["args"], args):
            local_state["variables"][param] = arg
        return self._execute_body(func["body"], output, local_state)

    def _execute_body(self, body: Dict, output: List[str], local_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Executes a body of AST nodes, optionally with local state."""
        state = local_state or self.state
        for node in body.get("nodes", []):
            result = self._process_node(node, output)
            if result.get("error"):
                return result
        return {"error": None}

    def _apply_pulse_data(self, pulse_data: Dict, output: List[str]):
        """Applies PulseSystem data to update execution state."""
        signal = pulse_data.get("signal")
        if signal == "update_state":
            component = pulse_data.get("component")
            new_state = pulse_data.get("state", {})
            if component in self.state["ui_state"]:
                self.state["ui_state"][component].update(new_state)
                output.append(f"State updated for {component}: {json.dumps(new_state)}")
                self.logger.log("pulse_state_update", signal.encode(), json.dumps(new_state).encode(),
                              {"component": component}, "INFO", "PULSE_UPDATE")
        elif signal == "update_fractal":
            fractal = pulse_data.get("fractal")
            new_params = pulse_data.get("params", {})
            if fractal in self.state["fractals"]:
                self.state["fractals"][fractal].update(new_params)
                output.append(f"Fractal updated: {fractal} with {json.dumps(new_params)}")
                self.logger.log("pulse_fractal_update", signal.encode(), json.dumps(new_params).encode(),
                              {"fractal": fractal}, "INFO", "PULSE_FRACTAL")

if __name__ == "__main__":
    # Example usage
    dsl_spec = {
        "language": "SovereignScript-v3.4",
        "symbols": {"assign": "=", "math": ["+", "-", "*", "/"], "end": [";"]},
        "structure": {
            "variable": {"declare": ["let <name> = <value>;"]},
            "output": ["print <value>;"],
            "function": {"define": ["fn <name>(<args>)"], "call": ["call <name>(<args>);"], "end": ["end"]},
            "control": {"if": ["if <condition>"], "loop": ["loop while <condition>"], "end": ["end"]},
            "ui": {"component": ["component <name> { <body> }"], "state": ["state <name> = <value>;"]},
            "fractal": {"define": ["fractal <name> = { <params> }"]}
        }
    }
    interpreter = SovereignScriptInterpreter(dsl_spec)
    code = """
    let x = 42;
    print x;
    fn add(a, b)
        let sum = a + b;
        print sum;
    end
    call add(10, 20);
    component Counter { state count = 0; }
    fractal Sierpinski = { iterations: 5, pattern: sierpinski };
    """
    result = interpreter.execute(code)
    print(json.dumps(result, indent=2))
