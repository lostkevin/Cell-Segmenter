from typing import Callable, Iterable, Optional
from tqdm import tqdm
import multiprocess

class TaskRunner:
    """
        对于给定的任务, 使用多线程执行命令
    
        可能需要的输入:
            worker function
            arguments iterator
            Pool arguments
    """
    def __init__(self, pool_size=8, cuda_context=False) -> None:
        if cuda_context:
            import torch
            # try:
            #     torch.multiprocessing.set_start_method('spawn')
            # except:
            #     if torch.multiprocessing.get_start_method() != 'spawn':
            #         raise SystemError()

            self.pool = torch.multiprocessing.Pool(pool_size)
            self._local_support = False
        else:
            self.pool = multiprocess.Pool(pool_size)
            self._local_support = True
        pass

    @staticmethod
    def __wrapper(func: Callable):
        def _wfunc(zipped_kwargs):
            return func(**zipped_kwargs)
        return _wfunc

    def run(self, func: Callable, args_iter: Iterable, *, total:Optional[int]=None):
        results = list(tqdm(self.pool.imap(self.__wrapper(func), args_iter), total=total))
        return results
    
    def map(self, func: Callable, args_iter: Iterable, *, total:Optional[int]=None):
        results = list(tqdm(self.pool.imap(func, args_iter), total=total))
        return results
    
    def terminate(self):
        self.pool.terminate()
        self.pool.join()
    
    @property
    def local(self):
        return self._local_support