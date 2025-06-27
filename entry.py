import os
import subprocess

def run_project():
    # 设置项目根目录
    project_root = "/home/nn/workspace/DiagFusion"
    
    try:
        # 首先激活.venv环境并运行debug.py
        debug_path = os.path.join(project_root, "DataProcess", "get_data.py")
        print("正在进行数据预处理...")
        debug_cmd = f". /home/nn/workspace/DiagFusion/dataprocess/bin/activate && python {debug_path}"
        subprocess.run(debug_cmd, shell=True, check=True)
        
        # 然后激活diagfusion环境并运行main.py
        print("\n正在运行DiagFusion算法...")
        main_cmd = f". /home/nn/workspace/DiagFusion/diagfusion/bin/activate && python main.py --config gaia_config2.yaml"
        subprocess.run(main_cmd, shell=True, cwd=project_root, check=True)
        
        print("\n所有任务执行完成!")
        
    except subprocess.CalledProcessError as e:
        print(f"执行过程中发生错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    run_project()
