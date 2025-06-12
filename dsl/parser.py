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

class DSLParser:
    def __init__(self, dsl_spec: Dict[str, Any]):
        self.spec = dsl_spec
        self.logger = TCCLogger()  # Assumes TCCLogger from provided context

    def parse(self, code: str) -> DSLNode:
        lines = code.strip().split('\n')
        root = DSLNode(type='program', value=None)
        current_block: List[DSLNode] = root.children
        stack: List[List[DSLNode]] = [current_block]
        line_number = 0

        for line in lines:
            line_number += 1
            line = line.strip()
            if not line:
                continue
            node = self._parse_line(line, line_number)
            if node:
                if node.type in ('function', 'component', 'if', 'loop', 'try', 'fractal', 'event'):
                    stack[-1].append(node)
                    stack.append(node.children)
                elif node.type == 'end':
                    if len(stack) > 1:
                        stack.pop()
                    else:
                        self.logger.log("parse_error", line.encode(), b"Unmatched end", {"line": line, "line_number": line_number}, "ERROR", "UNMATCHED_END")
                else:
                    stack[-1].append(node)

        if len(stack) > 1:
            self.logger.log("parse_error", b"", b"Unclosed block", {"line_number": line_number}, "ERROR", "UNCLOSED_BLOCK")

        return root

    def _parse_line(self, line: str, line_number: int) -> Optional[DSLNode]:
        # Variable declarations and updates
        for pattern in self.spec['structure']['variable']['declare']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)").replace("<value>", r"(.+?);"))}$', line)
            if match:
                name, value = match.groups()
                return DSLNode(type='variable', value={'name': name, 'value': value.strip(';')}, metadata={'action': 'declare', 'line': line_number})

        for pattern in self.spec['structure']['update']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)").replace("<value>", r"(.+?);"))}$', line)
            if match:
                name, value = match.groups()
                return DSLNode(type='variable', value={'name': name, 'value': value.strip(';')}, metadata={'action': 'update', 'line': line_number})

        # Output
        for pattern in self.spec['output']:
            match = re.match(rf'^{re.escape(pattern.replace("<value>", r"(.+?);"))}$', line)
            if match:
                return DSLNode(type='output', value=match.group(1).strip(';'), metadata={'line': line_number})

        # Function definition
        for pattern in self.spec['function']['define']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)").replace("<args>", r"(\w*(?:,\w*)*)"))}\s*{{?$', line)
            if match:
                name, args = match.groups()
                return DSLNode(type='function', value={'name': name, 'args': args.split(',') if args else []}, children=[], metadata={'line': line_number})

        # Function call
        for pattern in self.spec['function']['call']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)").replace("<args>", r"(.+?);"))}$', line)
            if match:
                name, args = match.groups()
                return DSLNode(type='call', value={'name': name, 'args': args.strip(';').split(',')}, metadata={'line': line_number})

        # Return
        for pattern in self.spec['function']['return']:
            match = re.match(rf'^{re.escape(pattern.replace("<value>", r"(.+?);"))}$', line)
            if match:
                return DSLNode(type='return', value=match.group(1).strip(';'), metadata={'line': line_number})

        # Control structures
        for pattern in self.spec['control']['if']:
            match = re.match(rf'^{re.escape(pattern.replace("<condition>", r"(.+?)"))}\s*{{?$', line)
            if match:
                return DSLNode(type='if', value=match.group(1), children=[], metadata={'line': line_number})

        for pattern in self.spec['control']['loop']:
            match = re.match(rf'^{re.escape(pattern.replace("<condition>", r"(.+?)"))}\s*{{?$', line)
            if match:
                return DSLNode(type='loop', value=match.group(1), children=[], metadata={'line': line_number})

        # Event handlers
        for pattern in self.spec['event']['hook']:
            match = re.match(rf'^{re.escape(pattern.replace("<event>", r"(\w+)").replace("<name>", r"(\w+)"))}\(\)\s*{{?$', line)
            if match:
                event, name = match.groups()
                return DSLNode(type='event', value={'event': event, 'handler': name}, children=[], metadata={'line': line_number})

        # UI Components
        for pattern in self.spec['ui']['component']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)").replace("<body>", r"(.+?)"))}\s*{{?$', line)
            if match:
                name, _ = match.groups()
                return DSLNode(type='component', value={'name': name}, children=[], metadata={'line': line_number})

        # CSS
        match = re.match(r'^(\w+)\s*\{\s*(.+?)\s*\}$', line)
        if match:
            selector, rules = match.groups()
            return DSLNode(type='css', value={'selector': selector, 'rules': rules}, metadata={'line': line_number})

        # Fractal definition
        for pattern in self.spec['fractal']['define']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)").replace("{ ... }", r"\{(.+?)\}"))}$', line)
            if match:
                name, body = match.groups()
                return DSLNode(type='fractal', value={'name': name, 'body': body}, children=[], metadata={'line': line_number})

        # Fractal render
        for pattern in self.spec['fractal']['render']:
            match = re.match(rf'^{re.escape(pattern.replace("<name>", r"(\w+)"))}\s*{{?$', line)
            if match:
                name = match.group(1)
                return DSLNode(type='fractal_render', value={'name': name}, metadata={'line': line_number})

        # End keywords
        if line in self.spec['function']['end'] + self.spec['control']['end'] + self.spec['errors']['end']:
            return DSLNode(type='end', value=line, metadata={'line': line_number})

        self.logger.log("parse_error", line.encode(), b"Invalid syntax", {"line": line, "line_number": line_number}, "ERROR", "SYNTAX")
        return None