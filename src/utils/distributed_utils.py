# src/utils/distributed_utils.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator

class DistributedTrainer:
    """分布式训练工具类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.accelerator = Accelerator()
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # 初始化分布式环境
        self._init_distributed()

    def _init_distributed(self):
        """初始化分布式环境"""
        if self.world_size > 1:
            dist.init_process_group(
                backend=self.config.get('backend', 'nccl'),
                init_method=self.config.get('init_method', 'env://')
            )
            torch.cuda.set_device(self.local_rank)

    def setup_model(self, model: torch.nn.Module) -> DDP:
        """设置分布式模型"""
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.get('find_unused_parameters', False)
            )
        return model

    def broadcast(self, data: Any) -> Any:
        """在分布式环境中广播数据"""
        if self.world_size <= 1:
            return data
        
        # 序列化数据
        tensor = torch.tensor(data, device=self.local_rank)
        dist.broadcast(tensor, src=0)
        return tensor

    def all_reduce(self, tensor: torch.Tensor, op: str = 'sum') -> torch.Tensor:
        """执行 All-Reduce 操作"""
        if self.world_size <= 1:
            return tensor
        
        op_map = {
            'sum': dist.ReduceOp.SUM,
            'mean': dist.ReduceOp.AVG,
            'max': dist.ReduceOp.MAX,
            'min': dist.ReduceOp.MIN
        }
        
        dist.all_reduce(tensor, op=op_map.get(op, dist.ReduceOp.SUM))
        return tensor

    def barrier(self):
        """执行分布式 Barrier"""
        if self.world_size > 1:
            dist.barrier()

    def cleanup(self):
        """清理分布式环境"""
        if self.world_size > 1:
            dist.destroy_process_group()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
