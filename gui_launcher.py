import webview
import sys
import threading
import time
import os
from app import app, init_system

# Ensure templates/static are found when frozen
if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app.template_folder = template_folder
    app.static_folder = static_folder

def start_server():
    # Initialize the QA system before starting server
    with app.app_context():
        init_system()
    # Run Flask app
    app.run(port=5000, use_reloader=False)

def main():
    # Start Flask in a separate thread
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()

    # Give Flask a moment to start
    time.sleep(1)

    # Create the window
    webview.create_window(
        "金融事件问答系统", 
        "http://127.0.0.1:5000",
        width=1200,
        height=800,
        resizable=True
    )
    
    # Start the GUI loop
    webview.start()

if __name__ == '__main__':
    main()
