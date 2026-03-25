import os
import sys
import wandb


class FileLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.terminal = sys.stdout
        self.file = open(path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


class WandbLogger:
    def __init__(self, enabled=False, project=None, entity=None, run_name=None, config=None, directory=None):
        self.enabled = False
        self.run = None
        if not enabled:
            return
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            dir=directory,
            config=config
        )
        self.enabled = True

    def log(self, metrics, step=None):
        if self.enabled and self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self):
        if self.enabled and self.run is not None:
            self.run.finish()
