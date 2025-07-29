import time
import psutil
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


class TorchProfiler:
    def __init__(self, use_profiler=True, output_dir="./logs/profiler", detailed_profiling=True):
        self.use_profiler = use_profiler
        self.output_dir = output_dir
        self.detailed_profiling = detailed_profiling
        self.profiler = None
        self.batch_start_time = None

    def __enter__(self):
        if self.use_profiler:
            if self.detailed_profiling:
                self.profiler = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=2, warmup=1, active=1, repeat=1), # Adjusted schedule for focus
                    on_trace_ready=tensorboard_trace_handler(self.output_dir),
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True
                )
            else:
                self.profiler = None
                print("Lightweight profiling: Only batch stats will be logged, no TensorBoard trace.")

            if self.profiler:
                self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.use_profiler and self.profiler:
            self.profiler.__exit__(exc_type, exc_value, traceback)

    def step(self):
        if self.use_profiler:
            if self.profiler:
                self.profiler.step()

    def log_batch_stats(self, batch_idx):
        if not torch.cuda.is_available():
            return
        if self.use_profiler:
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
            ram_usage = psutil.virtual_memory().percent
            batch_duration = time.time() - self.batch_start_time
            #print(f"[Batch {batch_idx}] GPU Mem: {gpu_mem:.2f} MB | RAM: {ram_usage:.1f}% | Time: {batch_duration:.3f} sec")

    def start_batch_timer(self):
        if self.use_profiler:
            self.batch_start_time = time.time()