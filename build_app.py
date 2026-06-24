import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

APP_VERSION = "v0.0.9"
RELEASE_DIR = "release"
RELEASE_FILENAME = f"FinancialQA-{APP_VERSION}-win-x64.zip"
REQUIRED_RUNTIME_PATHS = [
    "templates/index.html",
    "static/vendor/css/bootstrap.min.css",
    "static/vendor/js/bootstrap.bundle.min.js",
    "models/RE-GCN-master/data/80STOCKS/entity2id.txt",
    "models/RE-GCN-master/data/80STOCKS/relation2id.txt",
    "models/RE-GCN-master/data/80STOCKS/time2id.txt",
    "models/RE-GCN-master/data/80STOCKS/80stocks_quadruples.csv",
    "models/RE-GCN-master/models/80STOCKS-uvrgcn-convtranse-ly2-dilate1-his3-weight_0.5-discount_1.0-angle_10-dp0.2_0.2_0.2_0.2-gpu0",
    "models/TiRGN-main/data/80STOCKS/entity2id.txt",
    "models/TiRGN-main/data/80STOCKS/relation2id.txt",
    "models/TiRGN-main/data/80STOCKS/time2id.txt",
    "models/TiRGN-main/data/80STOCKS/history",
    "models/TiRGN-main/models/gl_rate_0.3-80STOCKS-convgcn-timeconvtranse-ly2-dilate1-his9-weight_0.5-discount_1.0-angle_14-dp0.2_0.2_0.2_0.2-gpu0-checkpoint",
]


def clean_build():
    """Clean previous build artifacts."""
    print("正在清理旧的构建目录...")
    for dir_name in ("build", "dist"):
        if os.path.exists(dir_name):
            print(f"  清理 {dir_name}")
            shutil.rmtree(dir_name, ignore_errors=True)


def ensure_build_dependencies():
    """Check whether packaging dependencies are available."""
    missing = []
    for module_name in ("PyInstaller", "webview"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        print("缺少打包依赖: " + ", ".join(missing))
        print("请先在当前 Python 环境中安装后再打包。")
        return False
    return True


def ensure_runtime_assets():
    """Validate files that must be bundled for the desktop exe to run alone."""
    missing = [path for path in REQUIRED_RUNTIME_PATHS if not Path(path).exists()]
    if missing:
        print("缺少独立运行所需资源，无法打包 exe:")
        for path in missing:
            print(f"  - {path}")
        return False
    return True


def build_app():
    """Run PyInstaller with the existing spec file."""
    print("开始执行 PyInstaller 打包...")
    cmd = [sys.executable, "-m", "PyInstaller", "FinancialQA.spec", "--clean", "--noconfirm"]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"打包失败，返回码: {result.returncode}")
        return False
    print("打包完成。")
    return True


def create_release_zip():
    """Create a zip archive for the packaged desktop application."""
    source_dir = os.path.join("dist", "FinancialQA")
    if not os.path.exists(source_dir):
        print("未找到 dist/FinancialQA，无法生成发布包。")
        return False

    os.makedirs(RELEASE_DIR, exist_ok=True)
    zip_filename = os.path.abspath(os.path.join(RELEASE_DIR, RELEASE_FILENAME))
    print(f"生成发布包: {zip_filename}")

    if os.path.exists(zip_filename):
        os.remove(zip_filename)

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                arc_name = os.path.relpath(file_path, "dist")
                zipf.write(file_path, arc_name)

    size_mb = os.path.getsize(zip_filename) / 1024 / 1024
    print(f"发布包已生成，大小约 {size_mb:.2f} MB")
    return True


def main():
    print("=" * 60)
    print("金融事件问答系统 - 打包工具")
    print("=" * 60)

    if not ensure_build_dependencies():
        sys.exit(1)

    if not ensure_runtime_assets():
        sys.exit(1)

    if not os.path.exists("FinancialQA.spec"):
        print("未找到 FinancialQA.spec，无法继续。")
        sys.exit(1)

    clean_build()

    if not build_app():
        sys.exit(1)

    if create_release_zip():
        print("打包与封装已完成。")
        print("输出目录: dist/FinancialQA")
        print(f"发布压缩包: {os.path.join(RELEASE_DIR, RELEASE_FILENAME)}")
    else:
        print("打包完成，但发布压缩包生成失败。")
        sys.exit(1)


if __name__ == "__main__":
    main()
