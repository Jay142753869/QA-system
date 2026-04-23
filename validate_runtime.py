import json
import subprocess
import sys
import time
import urllib.request


PYTHON_EXE = sys.executable
BASE_URL = "http://127.0.0.1:5000"
STATUS_URL = f"{BASE_URL}/api/status"
QUERY_URL = f"{BASE_URL}/api/query"


def http_json(url, payload=None, timeout=120):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_until_ready(max_wait_seconds=120):
    start = time.time()
    last_status = None
    while time.time() - start < max_wait_seconds:
        try:
            last_status = http_json(STATUS_URL, timeout=10)
            if last_status.get("ready") is True:
                return last_status
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Service did not become ready in time. Last status: {last_status}")


def main():
    server = subprocess.Popen([PYTHON_EXE, "app.py"])
    try:
        status = wait_until_ready()
        print(json.dumps({"status": status}, ensure_ascii=False))

        test_cases = [
            {"question": "贵州茅台的大股东是谁", "mode": "internal", "top_k": 3},
            {"question": "招商银行的董事长是谁", "mode": "internal", "top_k": 3},
            {"question": "招商银行的董事长是谁", "mode": "external", "top_k": 3},
        ]

        for case in test_cases:
            response = http_json(QUERY_URL, payload=case)
            summary = {
                "question": case["question"],
                "mode": case["mode"],
                "graph_result": response.get("graph_result"),
                "graph_message": response.get("graph_message"),
                "reasoning_result": response.get("reasoning_result"),
            }
            print(json.dumps(summary, ensure_ascii=False))
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
