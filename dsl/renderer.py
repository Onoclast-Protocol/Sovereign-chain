
import json
import hashlib
from typing import Dict, List, Any, Optional
from .parser import DSLParser
from .onoclast_chain import OnoclastChain
from .pulse_system import PulseSystem
from ..agents.tcc_logger import TCCLogger

class SovereignScriptRenderer:
    """Renders SovereignScript-v3.4 UI components and fractals for Onoclast Protocol IDE."""
    
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
        self.chain = chain
        self.pulse_system = pulse_system
        self.ui_components: Dict[str, Dict] = {}  # Cache for UI components
        self.fractal_configs: Dict[str, Dict] = {}  # Cache for fractal configs
        self.supported_fractals = {"sierpinski", "mandelbrot", "julia"}

    def render(self, code: str, pulse_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Renders SovereignScript code as HTML/CSS/JS for UI or Canvas for fractals."""
        try:
            # Security validation
            if self.chain:
                is_anomaly, score, explanation = self.chain.security.detect_anomaly(code.encode())
                if is_anomaly:
                    self.logger.log("render_anomaly", code.encode(), explanation.encode(),
                                  {"score": score}, "ERROR", "ANOMALY")
                    return {"html": "", "js": "", "error": f"Anomaly detected: {explanation}"}

            # Parse code to AST
            ast = self.parser.parse(code)
            self.logger.log("parse_ast", code.encode(), json.dumps(ast).encode(),
                          {"action": "render_parse"})

            # Initialize output
            html: List[str] = []
            js: List[str] = []
            component_id = f"comp_{hashlib.sha256(code.encode()).hexdigest()[:8]}"

            # Process AST nodes
            for node in ast.get("nodes", []):
                node_type = node.get("type")
                if node_type == "ui_component":
                    self._render_ui_component(node, html, js, component_id)
                elif node_type == "fractal_define":
                    self._render_fractal(node, html, js, component_id)
                elif node_type == "ui_eventHandlers":
                    self._render_event_handler(node, js, component_id)

            # Apply pulse data for dynamic updates
            if pulse_data and self.pulse_system:
                self._apply_pulse_data(pulse_data, js, component_id)

            # Combine output
            html_output = "\n".join(html) or "<div>No renderable content</div>"
            js_output = "\n".join(js) or "// No dynamic behavior"

            self.logger.log("render_success", code.encode(), html_output.encode(),
                          {"component_id": component_id})
            return {"html": html_output, "js": js_output, "error": None}

        except Exception as e:
            self.logger.log("render_error", code.encode(), str(e).encode(),
                          {"error": str(e)}, "ERROR", "RENDER")
            return {"html": "", "js": "", "error": str(e)}

    def _render_ui_component(self, node: Dict, html: List[str], js: List[str], component_id: str):
        """Renders a UI component node to HTML/CSS/JS."""
        name = node.get("name", "unnamed")
        props = node.get("props", {})
        state = node.get("state", {})
        body = node.get("body", {"nodes": []})
        comp_id = f"{component_id}_{name}"

        # Fetch on-chain assets if specified
        if self.chain and props.get("asset_cid"):
            try:
                asset_data = self.chain.ipfs_client.cat(props["asset_cid"]).decode()
                props["asset_url"] = f"data:image/png;base64,{asset_data}"
            except Exception as e:
                self.logger.log("asset_fetch_error", comp_id.encode(), str(e).encode(),
                              {"cid": props["asset_cid"]}, "ERROR", "ASSET_FETCH")

        # Generate HTML
        html.append(f'<div id="{comp_id}" class="ss-component">')
        for child in body.get("nodes", []):
            if child.get("type") == "ui_element":
                tag = child.get("tag", "div")
                attrs = child.get("attributes", {})
                content = child.get("content", "")
                attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
                html.append(f'<{tag} {attr_str}>{content}</{tag}>')
        html.append("</div>")

        # Generate CSS
        styles = node.get("styles", {})
        css = []
        for selector, rules in styles.items():
            css_rules = "; ".join(f"{k}: {v}" for k, v in rules.items())
            css.append(f".{comp_id} {selector} {{ {css_rules} }}")
        if css:
            html.append(f'<style>{" ".join(css)}</style>')

        # Generate JS for state and props
        js.append(f"const {comp_id}_state = {json.dumps(state)};")
        js.append(f"const {comp_id}_props = {json.dumps(props)};")
        js.append(f"function render_{comp_id}() {{")
        js.append(f"  const el = document.getElementById('{comp_id}');")
        js.append(f"  el.innerHTML = `{html[-2]}`;")  # Re-render inner content
        js.append("}")
        self.ui_components[comp_id] = {"state": state, "props": props}

    def _render_fractal(self, node: Dict, html: List[str], js: List[str], component_id: str):
        """Renders a fractal definition to Canvas-based JS."""
        name = node.get("name", "fractal")
        params = node.get("params", {})
        pattern = params.get("pattern", "sierpinski").lower()
        if pattern not in self.supported_fractals:
            self.logger.log("fractal_unsupported", name.encode(), pattern.encode(),
                          {"pattern": pattern}, "ERROR", "FRACTAL")
            return

        iterations = params.get("iterations", 5)
        scale = params.get("scale", 1.0)
        animate = params.get("animate", False)
        canvas_id = f"{component_id}_{name}_canvas"

        html.append(f'<canvas id="{canvas_id}" width="400" height="400"></canvas>')

        # Generate JS for fractal rendering
        js.append(f"""
        const {canvas_id} = document.getElementById('{canvas_id}');
        const ctx = {canvas_id}.getContext('2d');
        let animationFrameId;
        function draw_{name}(t = 0) {{
            ctx.clearRect(0, 0, {canvas_id}.width, {canvas_id}.height);
            ctx.beginPath();
            {self._generate_fractal_pattern(pattern, iterations, scale, animate)}
            ctx.stroke();
            if ({'true' if animate else 'false'}) {{
                animationFrameId = requestAnimationFrame((time) => draw_{name}(time / 1000));
            }}
        }}
        draw_{name}();
        function stop_{name}() {{
            cancelAnimationFrame(animationFrameId);
        }}
        """)
        self.fractal_configs[f"{component_id}_{name}"] = {
            "pattern": pattern, "iterations": iterations, "scale": scale, "animate": animate
        }

    def _generate_fractal_pattern(self, pattern: str, iterations: int, scale: float, animate: bool) -> str:
        """Generates JS code for a fractal pattern."""
        if pattern == "sierpinski":
            return f"""
            function sierpinski(x, y, size, level) {{
                if (level <= 0) return;
                ctx.moveTo(x, y);
                ctx.lineTo(x + size, y);
                ctx.lineTo(x + size / 2, y - size * Math.sqrt(3) / 2);
                ctx.lineTo(x, y);
                sierpinski(x, y, size / 2, level - 1);
                sierpinski(x + size / 2, y, size / 2, level - 1);
                sierpinski(x + size / 4, y - size * Math.sqrt(3) / 4, size / 2, level - 1);
            }}
            sierpinski(100 * {scale}, 300 * {scale}, 200 * {scale}, {iterations});
            """

        elif pattern == "mandelbrot":
            return f"""
            function mandelbrot() {{
                const w = {400 * scale}, h = {400 * scale};
                const maxIter = {iterations * 10};
                const zoom = {1.0 / scale};
                const moveX = {'Math.sin(t) * 0.1' if animate else '0'};
                const moveY = {'Math.cos(t) * 0.1' if animate else '0'};
                for (let x = 0; x < w; x++) {{
                    for (let y = 0; y < h; y++) {{
                        let cRe = (x - w / 2) / (w / 4) * zoom + moveX - 0.5;
                        let cIm = (y - h / 2) / (h / 4) * zoom + moveY;
                        let zRe = 0, zIm = 0;
                        let i = 0;
                        while (i < maxIter && zRe * zRe + zIm * zIm < 4) {{
                            let tmp = zRe * zRe - zIm * zIm + cRe;
                            zIm = 2 * zRe * zIm + cIm;
                            zRe = tmp;
                            i++;
                        }}
                        ctx.fillStyle = i === maxIter ? '#000' : `hsl(${{i / maxIter * 360}}, 100%, 50%)`;
                        ctx.fillRect(x, y, 1, 1);
                    }}
                }}
            }}
            mandelbrot();
            """

        elif pattern == "julia":
            return f"""
            function julia() {{
                const w = {400 * scale}, h = {400 * scale};
                const maxIter = {iterations * 10};
                const zoom = {1.0 / scale};
                const cRe = {'0.285 + Math.sin(t) * 0.01' if animate else '0.285'};
                const cIm = {'0.01 + Math.cos(t) * 0.01' if animate else '0.01'};
                for (let x = 0; x < w; x++) {{
                    for (let y = 0; y < h; y++) {{
                        let zRe = (x - w / 2) / (w / 4) * zoom;
                        let zIm = (y - h / 2) / (h / 4) * zoom;
                        let i = 0;
                        while (i < maxIter && zRe * zRe + zIm * zIm < 4) {{
                            let tmp = zRe * zRe - zIm * zIm + cRe;
                            zIm = 2 * zRe * zIm + cIm;
                            zRe = tmp;
                            i++;
                        }}
                        ctx.fillStyle = i === maxIter ? '#000' : `hsl(${{i / maxIter * 360}}, 100%, 50%)`;
                        ctx.fillRect(x, y, 1, 1);
                    }}
                }}
            }}
            julia();
            """
        return "// Unsupported fractal pattern"

    def _render_event_handler(self, node: Dict, js: List[str], component_id: str):
        """Renders event handlers for UI components."""
        event = node.get("event", "").lower()
        action = node.get("action", "")
        target = node.get("target", "")
        if event in ["onclick", "onhover", "oninput"]:
            js.append(f"""
            document.getElementById('{component_id}_{target}')?.addEventListener('{event[2:]}', () => {{
                {action}
                render_{component_id}_{target}();
            }});
            """)

    def _apply_pulse_data(self, pulse_data: Dict, js: List[str], component_id: str):
        """Applies PulseSystem data to update UI/fractal state."""
        signal = pulse_data.get("signal")
        target_id = pulse_data.get("component_id", component_id)

        if signal == "update_state":
            new_state = pulse_data.get("state", {})
            js.append(f"""
            Object.assign({target_id}_state, {json.dumps(new_state)});
            render_{target_id}();
            """)
            if target_id in self.ui_components:
                self.ui_components[target_id]["state"].update(new_state)
            self.logger.log("pulse_state_update", signal.encode(), json.dumps(new_state).encode(),
                          {"component_id": target_id}, "INFO", "PULSE_UPDATE")

        elif signal == "update_fractal":
            new_params = pulse_data.get("params", {})
            if target_id in self.fractal_configs:
                self.fractal_configs[target_id].update(new_params)
                js.append(f"""
                stop_{target_id.split('_')[1]}();
                draw_{target_id.split('_')[1]}();
                """)
                self.logger.log("pulse_fractal_update", signal.encode(), json.dumps(new_params).encode(),
                              {"component_id": target_id}, "INFO", "PULSE_FRACTAL")

if __name__ == "__main__":
    # Example usage
    dsl_spec = {
        "language": "SovereignScript-v3.4",
        "structure": {
            "ui": {
                "component": ["component <name> { <body> }"],
                "eventHandlers": ["onClick", "onHover", "onInput"]
            },
            "fractal": {
                "define": ["fractal <name> = { iterations: <value>, scale: <value>, pattern: <value> }"]
            }
        }
    }
    renderer = SovereignScriptRenderer(dsl_spec)
    code = """
    component Button {
        <button id="btn">Click me</button>
        onClick: alert('Clicked!');
        styles: { button { color: blue; } }
        state: { count: 0 }
    }
    fractal Mandelbrot = { iterations: 5, scale: 1.0, pattern: mandelbrot, animate: true }
    """
    result = renderer.render(code)
    print(json.dumps(result, indent=2))
