import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class DSLNode:
    type: str
    value: Any
    children: List['DSLNode'] = None
    metadata: Dict[str, Any] = None

    def __init__(self, type: str, value: Any, children: Optional[List['DSLNode']] = None, metadata: Optional[Dict[str, Any]] = None):
        self.type = type
        self.value = value
        self.children = children or []
        self.metadata = metadata or {}

class SwarmDSLRuntime:
    def __init__(self, dsl_spec: Dict[str, Any]):
        self.spec = dsl_spec
        self.variables: Dict[str, Any] = {}
        self.functions: Dict[str, DSLNode] = {}
        self.components: Dict[str, DSLNode] = {}
        self.event_handlers: Dict[str, List[DSLNode]] = {}
        self.logger = TCCLogger()  # Assumes TCCLogger from provided context

    def parse(self, code: str) -> DSLNode:
        lines = code.strip().split('\n')
        root = DSLNode(type='program', value=None)
        current_block: List[DSLNode] = root.children
        stack: List[List[DSLNode]] = [current_block]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            node = self._parse_line(line)
            if node:
                if node.type in ('function', 'component', 'if', 'loop', 'try', 'fractal'):
                    stack[-1].append(node)
                    stack.append(node.children)
                elif node.type in self.spec['function']['end'] + self.spec['control']['end'] + self.spec['errors']['end']:
                    stack.pop()
                else:
                    stack[-1].append(node)
        
        return root

    def _parse_line(self, line: str) -> Optional[DSLNode]:
        line = line.strip()
        
        # Variable declarations and updates
        for pattern in self.spec['structure']['variable']['declare'] + self.spec['structure']['update']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)").replace("<value>", r"(.+?);"))}$', line)
            if match:
                name, value = match.groups()
                return DSLNode(type='variable', value={'name': name, 'value': value.strip(';')}, metadata={'action': 'declare' if pattern in self.spec['structure']['variable']['declare'] else 'update'})
        
        # Output
        for pattern in self.spec['output']:
            match = re.match(rf'^{re.escape(pattern.replace("<value>", r"(.+?);"))}$', line)
            if match:
                return DSLNode(type='output', value=match.group(1).strip(';'))
        
        # Function definition
        for pattern in self.spec['function']['define']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)").replace("<args>", r"(\w*(?:,\w*)*)"))}\s*{{?$', line)
            if match:
                name, args = match.groups()
                return DSLNode(type='function', value={'name': name, 'args': args.split(',') if args else []}, children=[])
        
        # Function call
        for pattern in self.spec['function']['call']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)").replace("<args>", r"(.+?);"))}$', line)
            if match:
                name, args = match.groups()
                return DSLNode(type='call', value={'name': name, 'args': args.strip(';').split(',')})
        
        # Return
        for pattern in self.spec['function']['return']:
            match = re.match(rf'^{re.escape(pattern.replace("<value>", r"(.+?);"))}$', line)
            if match:
                return DSLNode(type='return', value=match.group(1).strip(';'))
        
        # Control structures
        for pattern in self.spec['control']['if']:
            match = re.match(rf'^{re.escape(pattern.replace("<condition>", r"(.+?)"))}\s*{{?$', line)
            if match:
                return DSLNode(type='if', value=match.group(1), children=[])
        
        for pattern in self.spec['control']['loop']:
            match = re.match(rf'^{re.escape(pattern.replace("<condition>", r"(.+?)"))}\s*{{?$', line)
            if match:
                return DSLNode(type='loop', value=match.group(1), children=[])
        
        # Event handlers
        for pattern in self.spec['event']['hook']:
            match = re.match(rf'^{re.escape(pattern.replace("<event>", r"(\w+)").replace("<name>", r"(\w+)"))}\(\)\s*{{?$', line)
            if match:
                event, name = match.groups()
                return DSLNode(type='event', value={'event': event, 'handler': name}, children=[])
        
        # UI Components
        for pattern in self.spec['ui']['component']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)").replace("<body>", r"(.+?)"))}\s*{{?$', line)
            if match:
                name, _ = match.groups()
                return DSLNode(type='component', value={'name': name}, children=[])
        
        # CSS
        match = re.match(r'^(\w+)\s*\{\s*(.+?)\s*\}$', line)
        if match:
            selector, rules = match.groups()
            return DSLNode(type='css', value={'selector': selector, 'rules': rules})
        
        # End keywords
        if line in self.spec['function']['end'] + self.spec['control']['end'] + self.spec['errors']['end']:
            return DSLNode(type='end', value=line)
        
        self.logger.log("parse_error", line.encode(), b"Invalid syntax", {"line": line}, "ERROR", "SYNTAX")
        return None

    def execute(self, node: DSLNode) -> Any:
        if node.type == 'program':
            return [self.execute(child) for child in node.children]
        
        elif node.type == 'variable':
            name = node.value['name']
            value = self._evaluate_expression(node.value['value'])
            self.variables[name] = value
            self.logger.log("variable", name.encode(), str(value).encode(), {'action': node.metadata['action']})
            return value
        
        elif node.type == 'output':
            value = self._evaluate_expression(node.value)
            self.logger.log("output", node.value.encode(), str(value).encode())
            print(value)
            return value
        
        elif node.type == 'function':
            self.functions[node.value['name']] = node
            self.logger.log("function_def", node.value['name'].encode(), b"Defined")
            return None
        
        elif node.type == 'call':
            func = self.functions.get(node.value['name'])
            if func:
                args = [self._evaluate_expression(arg.strip()) for arg in node.value['args']]
                self.logger.log("function_call", node.value['name'].encode(), json.dumps(args).encode())
                return self._execute_function(func, args)
            self.logger.log("call_error", node.value['name'].encode(), b"Function not found", "ERROR", "NO_FUNC")
            return None
        
        elif node.type == 'return':
            value = self._evaluate_expression(node.value)
            self.logger.log("return", node.value.encode(), str(value).encode())
            return value
        
        elif node.type == 'if':
            condition = self._evaluate_expression(node.value)
            self.logger.log("if", node.value.encode(), str(condition).encode())
            if condition:
                return [self.execute(child) for child in node.children]
            return None
        
        elif node.type == 'loop':
            condition = self._evaluate_expression(node.value)
            results = []
            while condition:
                for child in node.children:
                    results.append(self.execute(child))
                condition = self._evaluate_expression(node.value)
            self.logger.log("loop", node.value.encode(), json.dumps(results).encode())
            return results
        
        elif node.type == 'event':
            event, handler = node.value['event'], node.value['handler']
            if event not in self.event_handlers:
                self.event_handlers[event] = []
            self.event_handlers[event].append(node)
            self.logger.log("event_hook", event.encode(), handler.encode())
            return None
        
        elif node.type == 'component':
            self.components[node.value['name']] = node
            self.logger.log("component_def", node.value['name'].encode(), b"Defined")
            return None
        
        elif node.type == 'css':
            self.logger.log("css", node.value['selector'].encode(), node.value['rules'].encode())
            return node.value
        
        return None

    def _evaluate_expression(self, expr: str) -> Any:
        # Handle variables
        if expr in self.variables:
            return self.variables[expr]
        
        # Handle math operations
        for op in self.spec['symbols']['math']:
            if op in expr:
                left, right = expr.split(op, 1)
                left_val = self._evaluate_expression(left.strip())
                right_val = self._evaluate_expression(right.strip())
                if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                    if op == '+': return left_val + right_val
                    if op == '-': return left_val - right_val
                    if op == '*': return left_val * right_val
                    if op == '/': return left_val / right_val if right_val != 0 else None
                self.logger.log("math_error", expr.encode(), b"Invalid operands", "ERROR", "MATH")
                return None
        
        # Handle literals
        try:
            if expr.isdigit():
                return int(expr)
            return float(expr) if '.' in expr else expr
        except ValueError:
            return expr

    def _execute_function(self, func: DSLNode, args: List[Any]) -> Any:
        # Bind arguments to function parameters
        for param, arg in zip(func.value['args'], args):
            self.variables[param] = arg
        
        result = None
        for child in func.children:
            result = self.execute(child)
            if child.type == 'return':
                break
        
        # Clean up arguments
        for param in func.value['args']:
            self.variables.pop(param, None)
        
        return result

    def handle_event(self, event: str, data: Any) -> List[Any]:
        results = []
        for handler in self.event_handlers.get(event, []):
            func = self.functions.get(handler.value['handler'])
            if func:
                results.append(self._execute_function(func, [data]))
        return results

    def run(self, code: str) -> Any:
        ast = self.parse(code)
        self.logger.log("run", code.encode(), b"Starting execution")
        return self.execute(ast)