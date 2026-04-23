import ctypes
import os
import sys
import threading
import time
import webbrowser

import webview


def start_server():
    """Start the Flask server used by the desktop shell."""
    from app import app

    if getattr(sys, "frozen", False):
        template_folder = os.path.join(sys._MEIPASS, "templates")
        static_folder = os.path.join(sys._MEIPASS, "static")
        app.template_folder = template_folder
        app.static_folder = static_folder

    app.run(port=5000, use_reloader=False)


def main():
    """Launch the desktop wrapper and fall back to the browser if needed."""
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    try:
        webview.create_window(
            "金融事件问答系统",
            "http://127.0.0.1:5000",
            width=1200,
            height=800,
            resizable=True,
            text_select=True,
        )
        webview.start()
    except Exception as exc:
        url = "http://127.0.0.1:5000"
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
