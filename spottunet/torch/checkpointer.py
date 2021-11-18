from typing import Sequence

from dpipe.train import Checkpoints
from dpipe.io import PathLike
from typing import Dict, Any, Union, Iterable, Sequence

class CheckpointsWithBest(Checkpoints):
    def __init__(self, base_path: PathLike, objects: Union[Iterable, Dict[PathLike, Any]]):
        super().__init__(base_path, objects)
        self.best_so_far = -1

    def _get_best_checkpoint_folder(self):
        return self.base_path / f'{self._checkpoint_prefix}_best'
    def save(self, iteration: int, train_losses: Sequence = None, metrics: dict = None):
        """Save the states of all tracked objects."""
        current_folder = self._get_checkpoint_folder(iteration)
        current_folder.mkdir(parents=True)
        self._save_to(current_folder)
        if metrics['sdice_score'] > self.best_so_far:
            best_folder = self._get_best_checkpoint_folder()
            best_folder.mkdir(parents=True,exist_ok=True)
            self._save_to(best_folder)
            self.best_so_far = metrics['sdice_score']
        if iteration:
            self._clear_checkpoint(iteration - 1)

    def best_model_ckpt(self):
        self._get_best_checkpoint_folder() / 'model.pth'