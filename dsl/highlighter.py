import re
from typing import List, Tuple, Dict, Any

class SovereignScriptHighlighter:
    def __init__(self, dsl_spec: Dict[str, Any]):
        self.spec = dsl_spec
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> List[Tuple[str, re.Pattern]]:
        patterns = [
            # Comments
            (r'comment', re.compile(r'//.*$')),
            # Strings
            (r'string', re.compile(r'"[^"]*"|\'[^\']*\'')),
            # Numbers
            (r'number', re.compile(r'\b\d+(\.\d+)?\b')),
            # Keywords
            (r'keyword', re.compile(r'\b(' + '|'.join([
                *(p.split(' ')[0] for p in self.spec['structure']['variable']['declare']),
                *(p.split(' ')[0] for p in self.spec['structure']['update']),
                *(p.split(' ')[0] for p in self.spec['output']),
                *(p.split(' ')[0] for p in self.spec['function']['define']),
                *(p.split(' ')[0] for p in self.spec['function']['return']),
                *(p.split(' ')[0] for p in self.spec['control']['if']),
                *self.spec['control']['else'],
                *(p.split(' ')[0] for p in self.spec['control']['loop']),
                *self.spec['control']['iterate'],
                *self.spec['control']['end'],
                *self.spec['errors']['try'],
                *self.spec['errors']['catch'],
                *self.spec['errors']['fail'],
                *(p.split(' ')[0] for p in self.spec['event']['hook']),
                *(p.split(' ')[0] for p in self.spec['fractal']['define']),
                *(p.split(' ')[0] for p in self.spec['fractal']['render']),
                *(p.split(' ')[0] for p in self.spec['ui']['component']),
                *self.spec['ui']['eventHandlers'],
                *self.spec['ui']['lifecycle'],
                *self.spec['style']['keywords'],
                *self.spec['motion']['up'],
                *self.spec['motion']['down'],
                *self.spec['motion']['left'],
                *self.spec['motion']['right'],
                *self.spec['integration']['api']['get'],
                *self.spec['integration']['api']['post'],
                *self.spec['integration']['api']['put'],
                *self.spec['integration']['api']['delete'],
                *(p.split(' ')[0] for p in self.spec['integration']['interfaces']['route']),
                *(p.split(' ')[0] for p in self.spec['integration']['interfaces']['mount']),
                *self.spec['integration']['interfaces']['init']
            ]) + r')\b')),
            # Symbols
            (r'symbol', re.compile(r'[' + re.escape(
                self.spec['symbols']['assign'] + ''.join(self.spec['symbols']['math']) +
                '|'.join(self.spec['symbols']['end']) + r'{}()[],.'
            ) + r']')),
            # Identifiers
            (r'identifier', re.compile(r'\b[a-zA-Z_]\w*\b')),
            # CSS Properties
            (r'css_property', re.compile(r'\b(' + '|'.join(self.spec['style']['keywords']) + r')\b')),
            # CSS Values
            (r'css_value', re.compile(r'\b(' + '|'.join(self.spec['style']['values']) + r')\b'))
        ]
        return patterns

    def tokenize(self, code: str) -> List[Tuple[str, str, int, int]]:
        tokens = []
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            pos = 0
            while pos < len(line):
                match = None
                token_type = None
                for t_type, pattern in self.patterns:
                    m = pattern.match(line, pos)
                    if m:
                        match = m
                        token_type = t_type
                        break
                if match:
                    token = match.group(0)
                    tokens.append((token_type, token, line_num, pos + 1))
                    pos = match.end()
                else:
                    tokens.append(('error', line[pos], line_num, pos + 1))
                    pos += 1
        return tokens

    def highlight(self, code: str) -> str:
        tokens = self.tokenize(code)
        html = []
        current_line = 1
        for token_type, token, line_num, col in tokens:
            if line_num > current_line:
                html.append('\n' * (line_num - current_line))
                current_line = line_num
            class_name = f'ss-{token_type}'
            escaped_token = token.replace('<', '&lt;').replace('>', '&gt;')
            html.append(f'<span class="{class_name}">{escaped_token}</span>')
        return ''.join(html)