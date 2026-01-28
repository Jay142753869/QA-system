import os
import sys
import subprocess
import time

# ==============================================================================
# 配置：Anaconda Python 路径
# ==============================================================================
# 如果您在普通 Python 环境中运行此脚本，但希望使用 Anaconda 环境（支持 CUDA/GPU）进行训练，
# 请在此处填入 Anaconda 环境中 python.exe 的绝对路径。
# 方法：在 Anaconda Prompt 中输入 "where python" 获取路径。
# 示例：r"C:\Users\Admin\anaconda3\python.exe" 或 r"D:\anaconda3\envs\myenv\python.exe"
# 如果保持为 None，将使用当前运行此脚本的 Python 解释器。
ANACONDA_PYTHON_PATH = r"C:\Users\Jay14\anaconda3\envs\regcn_clean\python.exe"
# ==============================================================================

def train():
    # Base directory for the model code
    base_dir = os.path.join(os.getcwd(), 'models', 'RE-GCN-master')
    src_dir = os.path.join(base_dir, 'src')
    
    # Ensure the model directory is in PYTHONPATH so imports work
    env = os.environ.copy()
    python_path = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{base_dir};{python_path}"
    
    # The main training script
    main_script = os.path.join(src_dir, 'main.py')
    
    if not os.path.exists(main_script):
        print(f"Error: Training script not found at {main_script}")
        return

    # Determine which Python interpreter to use
    python_exec = sys.executable
    if ANACONDA_PYTHON_PATH and os.path.exists(ANACONDA_PYTHON_PATH):
        python_exec = ANACONDA_PYTHON_PATH
        print(f"Using configured Anaconda Python: {python_exec}")
    elif ANACONDA_PYTHON_PATH:
        print(f"Warning: Configured Anaconda path not found: {ANACONDA_PYTHON_PATH}")
        print(f"Falling back to current Python: {python_exec}")
    else:
        print(f"Using current Python: {python_exec}")

    # User-specified parameters
    cmd = [
        python_exec, main_script,
        '-d', '80STOCKS',
        '--train-history-len', '3',
        '--test-history-len', '3',
        '--dilate-len', '1',
        '--lr', '0.001',
        '--n-layers', '2',
        '--evaluate-every', '1',
        '--gpu', '0',
        '--n-hidden', '200',
        '--self-loop',
        '--decoder', 'convtranse',
        '--encoder', 'uvrgcn',
        '--layer-norm',
        '--weight', '0.5',
        '--entity-prediction',
        '--relation-prediction',
        '--angle', '10',
        '--discount', '1',
        '--task-weight', '0.7',
        '--n-epochs', '5'
    ]
    
    print("="*50)
    print("Starting REGCN Training with User Parameters")
    print("="*50)
    print(f"Working Directory: {src_dir}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # Run the training script
        # We set cwd to src_dir because the code might use relative paths like "../data"
        process = subprocess.Popen(
            cmd, 
            cwd=src_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            print("\n" + "="*50)
            print("Training Completed Successfully!")
            print("="*50)
        else:
            print("\n" + "="*50)
            print(f"Training Failed with exit code {process.returncode}")
            print("="*50)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train()
