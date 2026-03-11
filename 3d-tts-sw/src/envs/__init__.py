"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

from importlib import import_module


def get_env_datasets(env_name: str, **kwargs):
    if env_name == 'AMC23' or env_name == 'AIME24':
        task_module = import_module(f"envs.MATH")
    else:
        task_module = import_module(f"envs.{env_name}")
    return task_module.get_train_test_dataset(env_name, **kwargs)


def get_default_query_str_builder(env_name: str, **kwargs):
    task_module = import_module(f"envs.{env_name}")

    def fn(problem_input: str, is_few_shot: bool, model_names: list):
        return task_module.Env.build_query_str(
            cot_task_desc=task_module.COT_TASK_DESC,
            cot_examples=task_module.COT_EXAMPLES,
            problem_format_str=task_module.PROBLEM_FORMAT_STR,
            problem_input=problem_input,
            is_few_shot=is_few_shot,
            model_names=model_names
        )

    return fn
