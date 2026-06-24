import atexit
import ctypes
import logging
import os
import sys
import threading
import time
import webbrowser

logger = logging.getLogger(__name__)

# Global handle to the Werkzeug server so we can shut it down cleanly.
_server = None
_server_lock = threading.Lock()


def start_server():
    """Start the Flask server used by the desktop shell.

    Uses ``werkzeug.serving.make_server`` with ``port=0`` so the OS
    picks a free port.  The actual port is stored in ``_server_port``
    for the GUI to use.
    """
    global _server
    from app import app
    from werkzeug.serving import make_server

    if getattr(sys, "frozen", False):
        template_folder = os.path.join(sys._MEIPASS, "templates")
        static_folder = os.path.join(sys._MEIPASS, "static")
        app.template_folder = template_folder
        app.static_folder = static_folder

    with _server_lock:
        _server = make_server("127.0.0.1", 0, app)
        _server_port.append(_server.server_port)
    _server.serve_forever()


# Shared container so the main thread can read the assigned port.
_server_port = []


def _get_server_port(timeout=10):
    """Wait until the server thread has assigned a port, then return it."""
    waited = 0
    while not _server_port and waited < timeout:
        time.sleep(0.1)
        waited += 0.1
    if not _server_port:
        raise RuntimeError("Flask server did not start within timeout.")
    return _server_port[0]


def shutdown_server():
    """Gracefully shut down the Werkzeug server and release resources."""
    global _server
    with _server_lock:
        if _server is not None:
            logger.info("Shutting down Flask server...")
            _server.shutdown()
            _server = None

    # Release model resources if ReasoningEngine was initialised.
    try:
        from app import state
        if state.reasoning_engine is not None:
            if state.reasoning_engine.regcn_model is not None:
                logger.info("Releasing REGCN model resources.")
                del state.reasoning_engine.regcn_model
                state.reasoning_engine.regcn_model = None
            if state.reasoning_engine.tirgn_model is not None:
                logger.info("Releasing TiRGN model resources.")
                del state.reasoning_engine.tirgn_model
                state.reasoning_engine.tirgn_model = None
    except Exception:
        pass

    # Close graph database driver.
    try:
        from app import state
        if state.graph_dao is not None:
            state.graph_dao.close()
            state.graph_dao = None
    except Exception:
        pass


def main():
    """Launch the desktop wrapper and fall back to the browser if needed."""
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Register cleanup on exit.
    atexit.register(shutdown_server)

    try:
        port = _get_server_port()
        url = f"http://127.0.0.1:{port}"
        logger.info("Server started on port %d", port)

        import webview

        window = webview.create_window(
            "金融事件问答系统",
            url,
            width=1200,
            height=800,
            resizable=True,
            text_select=True,
        )

        # Shut down the server when the window is closed.
        window.events.closed += shutdown_server
        webview.start()
    except Exception as exc:
        # Fallback: if the port was assigned, open browser; else use 5000.
        port = _server_port[0] if _server_port else 5000
        url = f"http://127.0.0.1:{port}"
        time.sleep(3)
        webbrowser.open(url)

        msg = (
            "内嵌窗口启动失败，已自动在浏览器中打开系统页面。\n\n"
            f"错误信息: {exc}\n\n"
            "请保持此提示框不要关闭，否则后台服务可能会停止。\n"
            f"浏览器访问地址: {url}"
        )
        ctypes.windll.user32.MessageBoxW(None, msg, "金融事件问答系统", 0x00000040)

        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
