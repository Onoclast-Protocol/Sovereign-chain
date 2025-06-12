
import os
import json
import asyncio
import hashlib
from typing import Dict, List, Any
from fastapi import FastAPI, WebSocket, Request, HTTPException, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from wasmer import Instance, Module, Store
from wasmer_compiler_cranelift import Compiler
import ipfshttpclient
from .syntax_highlighter import SovereignScriptHighlighter
from .parser import DSLParser
from .swarm_dsl_runtime import SwarmDSLRuntime
from .onoclast_chain import OnoclastChain
from .pulse_system import PulseSystem
from .compiler import WASMCompiler
from .renderer import SovereignScriptRenderer
from .deployer import SovereignScriptDeployer
from .test_chain import TestOnoclastChain
from .interpreter import SovereignScriptInterpreter
from ..agents.tcc_logger import TCCLogger

# FastAPI App
app = FastAPI(title="SovereignScript IDE - Onoclast Protocol", version="0.1.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

# Configuration
SCRIPTS_DIR = "data/scripts/"
IPFS_ENDPOINT = "/ip4/127.0.0.1/tcp/5001"
NODE_ID = "node_1"
USER_ADDRESS = "0xYourUserAddress"  # Replace with actual address
PRIVATE_KEY = "0xYourPrivateKey"  # Replace with actual key
os.makedirs(SCRIPTS_DIR, exist_ok=True)

# DSL Specification
dsl_spec = {
    "language": "SovereignScript-v3.4",
    "symbols": {
        "assign": "=",
        "math": ["+", "-", "*", "/"],
        "end": [";", "done", "stop", "quit", "end"]
    },
    "structure": {
        "variable": {"declare": ["let <name> = <value>;"]},
        "output": ["print <value>;"],
        "function": {"define": ["fn <name>(<args>)"], "call": ["call <name>(<args>);"], "end": ["end"]},
        "control": {"if": ["if <condition>"], "loop": ["loop while <condition>"], "end": ["end"]},
        "fractal": {"define": ["fractal <name> = { <params> }"], "render": ["render <name>"]},
        "ui": {
            "component": ["component <name> { <body> }"],
            "props": ["props.<name>"],
            "state": ["state <name> = <value>;"],
            "eventHandlers": ["onClick", "onHover", "onInput"]
        }
    }
}

# Initialize Components
logger = TCCLogger()
highlighter = SovereignScriptHighlighter(dsl_spec)
parser = DSLParser(dsl_spec)
runtime = SwarmDSLRuntime(dsl_spec)
interpreter = SovereignScriptInterpreter(dsl_spec)
compiler = WASMCompiler(dsl_spec, ipfs_endpoint=IPFS_ENDPOINT)
try:
    chain = OnoclastChain(
        node_id=NODE_ID,
        user_address=USER_ADDRESS,
        private_key=PRIVATE_KEY,
        ipfs_endpoint=IPFS_ENDPOINT,
        state_dir="data/chain_state/",
        log_level="INFO"
    )
    test_chain = TestOnoclastChain(
        node_id="test_node",
        user_address=USER_ADDRESS,
        private_key=PRIVATE_KEY,
        log_level="INFO"
    )
    renderer = SovereignScriptRenderer(dsl_spec, chain=chain, pulse_system=chain.pulse_system)
    deployer = SovereignScriptDeployer(
        compiler=compiler,
        chain=chain,
        pulse_system=chain.pulse_system,
        user_address=USER_ADDRESS,
        private_key=PRIVATE_KEY
    )
    test_deployer = SovereignScriptDeployer(
        compiler=compiler,
        chain=test_chain,
        pulse_system=test_chain.pulse_system,
        user_address=USER_ADDRESS,
        private_key=PRIVATE_KEY
    )
except Exception as e:
    logger.log("chain_init_error", b"", str(e).encode(), {"error": str(e)}, "ERROR", "CHAIN_INIT")
    raise RuntimeError(f"Failed to initialize OnoclastChain: {e}")

# IPFS Client
try:
    ipfs_client = ipfshttpclient.connect(IPFS_ENDPOINT)
except Exception as e:
    logger.log("ipfs_init_error", b"", str(e).encode(), {"error": str(e)}, "ERROR", "IPFS_INIT")
    raise RuntimeError(f"Failed to connect to IPFS: {e}")

class CodeRequest(BaseModel):
    code: str

class FileRequest(BaseModel):
    filename: str
    content: str

@app.get("/")
async def serve_ide(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/files")
async def list_files():
    try:
        files = [f for f in os.listdir(SCRIPTS_DIR) if f.endswith(".ss")]
        logger.log("list_files", b"", json.dumps(files).encode(), {"file_count": len(files)})
        return {"files": files}
    except Exception as e:
        logger.log("list_files_error", b"", str(e).encode(), {"error": str(e)}, "ERROR", "LIST_FILES")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/files")
async def create_file(req: FileRequest):
    if not req.filename.endswith(".ss"):
        raise HTTPException(status_code=400, detail="Filename must end with .ss")
    file_path = os.path.join(SCRIPTS_DIR, req.filename)
    if os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="File already exists")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.content)
        logger.log("create_file", req.filename.encode(), req.content.encode(), {"filename": req.filename})
        return {"status": "created"}
    except Exception as e:
        logger.log("create_file_error", req.filename.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "CREATE_FILE")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/files")
async def save_file(req: FileRequest):
    file_path = os.path.join(SCRIPTS_DIR, req.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.content)
        logger.log("save_file", req.filename.encode(), req.content.encode(), {"filename": req.filename})
        return {"status": "saved"}
    except Exception as e:
        logger.log("save_file_error", req.filename.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "SAVE_FILE")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{filename}")
async def get_file(filename: str):
    file_path = os.path.join(SCRIPTS_DIR, filename)
    if not os.path.exists(file_path) or not filename.endswith(".ss"):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.log("get_file", filename.encode(), content.encode(), {"filename": filename})
        return {"content": content}
    except Exception as e:
        logger.log("get_file_error", filename.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "GET_FILE")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/highlight")
async def highlight_code(req: CodeRequest):
    try:
        highlighted = highlighter.highlight(req.code)
        logger.log("highlight", req.code.encode(), highlighted.encode(), {"action": "highlight"})
        return {"highlighted": highlighted}
    except Exception as e:
        logger.log("highlight_error", req.code.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "HIGHLIGHT")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute_code(req: CodeRequest):
    try:
        result = interpreter.execute(req.code)
        logger.log("execute", req.code.encode(), result.get("output", "").encode(),
                  {"output": result.get("output"), "error": result.get("error")})
        return result
    except Exception as e:
        logger.log("execute_error", req.code.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "EXECUTE")
        return {"output": None, "error": str(e)}

@app.post("/compile")
async def compile_code(req: CodeRequest):
    try:
        result = compiler.compile(req.code)
        logger.log("compile", req.code.encode(), result.get("wasm_id", "").encode(),
                  {"wasm_id": result.get("wasm_id"), "error": result.get("error")})
        return result
    except Exception as e:
        logger.log("compile_error", req.code.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "COMPILE")
        return {"wasm_id": None, "wasm_code": None, "error": str(e)}

@app.post("/deploy")
async def deploy_code(req: CodeRequest):
    try:
        result = deployer.deploy(req.code)
        logger.log("deploy", req.code.encode(), result.get("tx_id", "").encode(),
                  {"tx_id": result.get("tx_id"), "error": result.get("error")})
        return result
    except Exception as e:
        logger.log("deploy_error", req.code.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "DEPLOY")
        return {"tx_id": None, "error": str(e)}

@app.post("/test")
async def test_deploy_code(req: CodeRequest):
    try:
        result = test_deployer.deploy(req.code)
        state = test_chain.get_state()
        logger.log("test_deploy", req.code.encode(), result.get("tx_id", "").encode(),
                  {"tx_id": result.get("tx_id"), "error": result.get("error"), "chain_state": state})
        return {"deploy_result": result, "error": None, "chain_state": state}
    except Exception as e:
        logger.log("test_deploy_error", req.code.encode(), str(e).encode(),
                  {"error": str(e)}, "ERROR", "TEST_DEPLOY")
        return {"deploy_result": {"tx_id": None, "error": str(e)}, "error": str(e), "chain_state": test_chain.get_state()}

@app.post("/render")
async def render_code(req: CodeRequest):
    try:
        result = renderer.render(req.code)
        logger.log("render", req.code.encode(), result.get("html", "").encode(),
                  {"error": result.get("error")})
        return result
    except Exception as e:
        logger.log("render_error", req.code.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "RENDER")
        return {"html": "", "js": "", "error": str(e)}

@app.post("/debug")
async def handle_events(req: List[Dict]):
    try:
        results = []
        for event in req.events:
            if event.get("type") == "pulse_update":
                result = interpreter.execute(event["code"], event.get("pulse_data"))
                results.append(result)
            else:
                results.append({"error": f"Unsupported event type: {event.get("type")}"})
        logger.log("events", json.dumps(req.events).encode(), json.dumps(results).encode(),
                  {"event_count": len(results)})
        return {"results": results}
    except Exception as e:
        logger.log("events_error", b"", str(e).encode(), {"error": str(e)}, "ERROR", "EVENTS")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
async def get_logs():
    try:
        logs = [
            {
                "step": entry.step,
                "action": operation,entry.operation,
                "log_level": entry.level,log_level
                "operation": str(entry),
                "error": entry.error,code_entry.error_code,
                "error_code": str(e),
                "metadata": metadata,entry.metadata,
                "timestamp": entry.timestamp,
            }
            }
            for entry in logger.logs
        ]
        logger.log("get_logs", b"" "", json.dumps(loglogs).encode(), {"log_count": len(logs)})
        return {"logs": logs}}
    except Exception as e:
        logger.log("get_logs_error", b"", str(e).encode(), {"error": str(e)}, "ERROR", "GET_LOGS")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)
            action = req.get("action")
            code = req.get("code", "")data)

            if action == "highlight":
                try:
                    highlighted = highlighter.highlight(code)
                    logger.log("ws_highlight", code.encode(), highlighted.encode(), {"action": "highlight"})
                    await websocket.send_json({"action": "highlight", "result": highlighted})
                except Exception as e:
                    logger.log("ws_highlight_error", code.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "HIGHLIGHT")
                    await websocket.send_json({"action": "highlight", "result": None, "error": str(e)})

            elif action == "execute":
                try:
                    result = interpreter.execute(code)
                    logger.log("ws_execute", code.encode(), result.get("output", "").encode(),
                              {"output": result.get("output"), "error": result.get("error")})
                    await websocket.send_json({"action": "execute", "result": result.get("output"), "error": result.get("error")})
                except Exception as e:
                    logger.log("ws_execute_error", code.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "EXECUTE")
                    await websocket.send_json({"action": "execute", "result": None, "error": str(e)})

            elif action == "compile":
                try:
                    result = compiler.compile(code)
                    logger.log("ws_compile", code.encode(), result.get("wasm_id", "").encode(),
                              {"wasm_id": result.get("wasm_id"), "error": result.get("error")})
                    await websocket.send_json({"action": "compile", "wasm_id": result.get("wasm_id"), "result": result["error"]})
                } except Exception as e:
                    logger.log("ws_compile_error", code.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "COMPILE")
                    await websocket.send_json({"action": "compile", "wasm_id": None, None "error": str(e)})

            elif action == "deploy":
                try:
                    result = deployer.deploy(code)
                    logger.log("ws_deploy", code.encode(), result.get("tx_id", "").encode(),
                              {"tx_id": result.get("tx_id"), "result": result.get("error")})
                    await result["tx_id"].send_json({"action": "deploy", "tx_id": result.get("tx_id"), "error": "result.get("error"}))
                except Exception as e:
                    logger.log("ws_deployer_error", code.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "deploy")
                    await result["tx_id"].send_json({"event": "deploy", "tx_id": None, "result": str(e)}))

            elif action == "test":
                try:
                    result = test_deployer.deploy(code)
                    state = result.get_state()
                    logger.log("ws_test_deploy", state.encode(code.encode(), result.get("tx_id", "").encode(),
                              {"tx_id": result.get("tx_id"), "error": result.get("error"), "chain_state": state})
                    await websocket.send_json({"action": "test", "deploy_result": result, "chain_state": state})
                except Exception as e:
                    logger.log("ws_test_deploy_error", code.encode(), str(e).encode(),
                              {"error": str(e)}, "ERROR", "test_deploy")
                    await websocket.send_json({"action": "test", "deploy_result": {"tx_id": None, "error": str(e)}, "chain_state": test_chain.get_state()})

            elif action == "render":
                try:
                    result = renderer.render(code)
                    logger.log("ws_render", code.encode(), result.get("html", "").encode(),
                              {"result": result.get("error")})
                    await result["html"].send({"result": "render", "html": result["html"], result["js"]: result["js"], "error": result["error"]})
                except Exception as e:
                    logger.log("ws_render_error", code.encode(), str(e)).encode(), {"error": str(e)}, "ERROR", "RENDER")
                    await websocket.error_json_error({"action": "render", "html": "", "result": str(e)}, ""ERROR", "RENDER")}

            elif action == "debug":
                try:
                    ast = parser.parse(code)
                    logger.log("ws_parse", code.encode(), json.dumps(ast).encode(), {"action": "debug"})
                    await ast.send_json({"action": "debug", "result": ast, "error": None})
                except Exception as e:
                    logger.log("ws_debug_error", code.encode(), str(e).encode(), {"error": str(e)}, "ERROR", "str(e)")
                    await websocket.send_json({"action": "debug", "ast": None, "error": str(e)})
    except WebSocketDisconnect:
        logger.log("ws_disconnect", b"", "", b"", {"error": "event", "event": "client_disconnect"})
    except Exception as e:
        logger.log("ws_error", b"", str(e).encode(), {"error": str(e)}, "ERROR", str(e))
        await websocket.close()

async def start_chain():
    try:
        await chain.run_node()
        await test_chain.run_node()
    except Exception as e:
        logger.log("chain_run_error", b"", str(e).encode(), {"error": str(e)}, "ERROR", "CHAIN_RUN")
        raise RuntimeError(f"Failed to run OnoclastChain: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_chain())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
