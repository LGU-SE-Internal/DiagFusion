#!/usr/bin/env python3
"""
系统性能监控工具
用于监控并行处理时的CPU、内存、磁盘IO等指标
"""

import psutil
import time
import threading
import json
import os
from datetime import datetime
from pathlib import Path
import signal
import sys
from collections import deque
import subprocess

class SystemMonitor:
    def __init__(self, interval=1.0, max_points=300):
        """
        Args:
            interval: 监控间隔(秒)
            max_points: 最大数据点数量
        """
        self.interval = interval
        self.max_points = max_points
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 数据存储
        self.timestamps = deque(maxlen=max_points)
        self.cpu_percent = deque(maxlen=max_points)
        self.memory_percent = deque(maxlen=max_points)
        self.memory_used_gb = deque(maxlen=max_points)
        self.disk_io_read = deque(maxlen=max_points)
        self.disk_io_write = deque(maxlen=max_points)
        self.process_count = deque(maxlen=max_points)
        self.load_avg = deque(maxlen=max_points)
        
        # 用于计算增量的前一次值
        self.prev_disk_io = None
        
        # 日志文件
        self.log_file = Path("system_monitor.log")
        
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            print("监控已在运行中")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"开始系统监控，间隔: {self.interval}秒")
        print("按 Ctrl+C 停止监控")
        
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("\n监控已停止")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                timestamp = datetime.now()
                
                # CPU 使用率 (总体和每核心)
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
                
                # 内存使用率
                memory = psutil.virtual_memory()
                
                # 磁盘IO
                disk_io = psutil.disk_io_counters()
                
                # 进程数量
                process_count = len(psutil.pids())
                
                # 系统负载
                try:
                    load_avg = os.getloadavg()[0]
                except (OSError, AttributeError):
                    load_avg = 0
                
                # 计算磁盘IO速率
                disk_read_rate = 0
                disk_write_rate = 0
                if self.prev_disk_io:
                    disk_read_rate = (disk_io.read_bytes - self.prev_disk_io.read_bytes) / self.interval / (1024*1024)  # MB/s
                    disk_write_rate = (disk_io.write_bytes - self.prev_disk_io.write_bytes) / self.interval / (1024*1024)  # MB/s
                
                # 存储数据
                self.timestamps.append(timestamp)
                self.cpu_percent.append(cpu_percent)
                self.memory_percent.append(memory.percent)
                self.memory_used_gb.append(memory.used / (1024**3))
                self.disk_io_read.append(disk_read_rate)
                self.disk_io_write.append(disk_write_rate)
                self.process_count.append(process_count)
                self.load_avg.append(load_avg)
                
                # 更新前一次值
                self.prev_disk_io = disk_io
                
                # 实时输出
                self._print_current_stats(timestamp, cpu_percent, cpu_per_core, memory, 
                                        disk_read_rate, disk_write_rate, 
                                        process_count, load_avg)
                
                # 写入日志
                self._write_log(timestamp, cpu_percent, memory.percent, 
                              memory.used / (1024**3), disk_read_rate, 
                              disk_write_rate, process_count, load_avg)
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(self.interval)
                
    def _print_current_stats(self, timestamp, cpu_percent, cpu_per_core, memory, 
                           disk_read_rate, disk_write_rate, process_count, load_avg):
        """打印当前统计信息"""
        # 清除当前行并打印新数据
        print(f"\r{timestamp.strftime('%H:%M:%S')} | "
              f"CPU: {cpu_percent:5.1f}% | "
              f"内存: {memory.percent:5.1f}% ({memory.used/(1024**3):5.2f}GB) | "
              f"磁盘: R{disk_read_rate:6.1f} W{disk_write_rate:6.1f} MB/s | "
              f"进程: {process_count:4d} | "
              f"负载: {load_avg:4.2f}", end="", flush=True)
        
        # 每10秒打印一次详细的CPU核心信息
        if timestamp.second % 10 == 0:
            print(f"\nCPU核心使用率: {[f'{core:.1f}%' for core in cpu_per_core]}")
        
    def _write_log(self, timestamp, cpu_percent, memory_percent, memory_gb,
                   disk_read_rate, disk_write_rate, process_count, load_avg):
        """写入日志文件"""
        log_data = {
            'timestamp': timestamp.isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_gb,
            'disk_read_mb_s': disk_read_rate,
            'disk_write_mb_s': disk_write_rate,
            'process_count': process_count,
            'load_avg': load_avg
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
            
    def get_process_info(self, name_filter=None, top_n=10):
        """获取进程信息"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                proc_info = proc.info
                if name_filter is None or name_filter.lower() in proc_info['name'].lower():
                    processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        # 按CPU使用率排序
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        return processes[:top_n]
    
    def print_system_info(self):
        """打印系统基本信息"""
        print("=== 系统信息 ===")
        print(f"CPU核心数: {psutil.cpu_count(logical=False)} 物理, {psutil.cpu_count(logical=True)} 逻辑")
        
        memory = psutil.virtual_memory()
        print(f"内存总量: {memory.total / (1024**3):.2f} GB")
        
        # 磁盘信息
        print("\n=== 磁盘使用情况 ===")
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                print(f"{partition.device}: {usage.used / (1024**3):.1f}GB / {usage.total / (1024**3):.1f}GB "
                      f"({usage.percent:.1f}%)")
            except PermissionError:
                continue
                
    def print_top_processes(self, n=10):
        """打印CPU使用率最高的进程"""
        print(f"\n=== CPU使用率最高的{n}个进程 ===")
        processes = self.get_process_info(top_n=n)
        
        print(f"{'PID':<8} {'进程名':<20} {'CPU%':<8} {'内存%':<8} {'状态':<10}")
        print("-" * 60)
        
        for proc in processes:
            print(f"{proc['pid']:<8} {proc['name'][:19]:<20} "
                  f"{proc['cpu_percent'] or 0:<8.1f} {proc['memory_percent'] or 0:<8.1f} "
                  f"{proc['status']:<10}")
    
    def show_live_dashboard(self):
        """显示实时仪表盘"""
        self.print_system_info()
        self.start_monitoring()
        
        def signal_handler(sig, frame):
            self.stop_monitoring()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            while self.is_monitoring:
                time.sleep(5)  # 每5秒更新一次进程信息
                print("\n")
                self.print_top_processes()
                print()
        except KeyboardInterrupt:
            self.stop_monitoring()


def check_system_tools():
    """检查系统可用的监控工具"""
    print("=== 系统监控工具检查 ===")
    
    tools = {
        'htop': 'htop --version',
        'top': 'top -v',
        'iotop': 'iotop --version',
        'iostat': 'iostat -V',
        'vmstat': 'vmstat -V',
        'free': 'free --version',
        'df': 'df --version',
        'ps': 'ps --version'
    }
    
    available_tools = []
    for tool, cmd in tools.items():
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                available_tools.append(tool)
                print(f"✅ {tool}: 可用")
            else:
                print(f"❌ {tool}: 不可用")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"❌ {tool}: 未安装")
    
    return available_tools


def install_htop():
    """安装htop系统工具"""
    print("=== 安装htop监控工具 ===")
    try:
        # 尝试使用不同的包管理器安装htop
        commands = [
            ['sudo', 'apt', 'install', '-y', 'htop'],
            ['sudo', 'yum', 'install', '-y', 'htop'], 
            ['sudo', 'dnf', 'install', '-y', 'htop'],
            ['brew', 'install', 'htop']
        ]
        
        for cmd in commands:
            try:
                print(f"尝试执行: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print("✅ htop 安装成功!")
                    return True
                else:
                    print(f"命令失败: {result.stderr}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
                
        print("❌ 无法自动安装htop，请手动安装")
        print("Ubuntu/Debian: sudo apt install htop")
        print("CentOS/RHEL: sudo yum install htop")
        print("macOS: brew install htop")
        return False
        
    except Exception as e:
        print(f"安装过程中出错: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="系统性能监控工具")
    parser.add_argument("--interval", "-i", type=float, default=1.0, help="监控间隔(秒)")
    parser.add_argument("--check-tools", action="store_true", help="检查可用的系统监控工具")
    parser.add_argument("--install-htop", action="store_true", help="尝试安装htop")
    parser.add_argument("--processes", "-p", type=int, default=10, help="显示进程数量")
    
    args = parser.parse_args()
    
    if args.check_tools:
        available_tools = check_system_tools()
        if 'htop' not in available_tools:
            print("\n建议安装htop以获得更好的监控体验")
            print("运行: python system_monitor.py --install-htop")
        sys.exit(0)
        
    if args.install_htop:
        install_htop()
        sys.exit(0)
    
    # 创建监控器
    monitor = SystemMonitor(interval=args.interval)
    
    print("DiagFusion 系统性能监控器")
    print("=" * 50)
    
    try:
        monitor.show_live_dashboard()
    except KeyboardInterrupt:
        print("\n监控已停止") 