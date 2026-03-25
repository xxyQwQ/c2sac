DEFAULT_WALKER_TASK_NAMES = ('walk', 'run')


def normalize_task_names(task_names):
    return [str(task_name) for task_name in task_names]


def validate_task_names(task_names):
    task_names = normalize_task_names(task_names)
    if not task_names:
        raise ValueError('At least one task name is required.')
    if len(set(task_names)) != len(task_names):
        raise ValueError(f'Task names must be unique, got: {task_names}')
    return task_names
