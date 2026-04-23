import os
import shutil
import subprocess
import sys
import zipfile

APP_VERSION = "v0.0.9"
RELEASE_DIR = "release"
RELEASE_FILENAME = f"FinancialQA-{APP_VERSION}-win-x64.zip"


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


def build_app():
    """Run PyInstaller with the existing spec file."""
    print("开始执行 PyInstaller 打包...")
    cmd = [sys.executable, "-m", "PyInstaller", "FinancialQA.spec", "--clean", "--noconfirm"]
    result = subprocess.run(cmd, check=False)
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
