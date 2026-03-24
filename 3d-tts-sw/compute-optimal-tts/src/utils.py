import argparse
import copy
import os
import json
import jsonlines
import pickle
import time
import threading
import torch
import torch.distributed as dist
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def get_model_name(model_name_or_path: str):
    model_name = model_name_or_path
    if model_name[-1] == '/':
        model_name = model_name[:-1]
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    if '--' in model_name:
        model_name = model_name.split('--')[-1]
    return model_name


def is_file_exists(file_path):
    return os.path.exists(file_path)


def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0


def create_empty_file(file_path):
    with open(file_path, 'w') as f:
        pass


def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        print(f"Load jsonl file error: {str(e)}")
        data = {}
    return data


def load_jsonl(file_path):
    data = []
    try:
        with jsonlines.open(file_path, "r") as reader:
            for obj in reader:
                data.append(obj)
    except Exception as e:
        print(f"Load jsonl file error: {str(e)}")
    return data


def get_jsonl_file_num(save_dir, q_idx):
    question_path = os.path.join(save_dir, f"question_{q_idx}")
    if not is_file_exists(question_path):
        return 0
    file_list = [file for file in os.listdir(question_path) if file.endswith(".jsonl")]
    return len(file_list)


def get_file_exist_time(file_path):
    current_time = time.time()
    try:
        creation_time = os.path.getmtime(file_path)
    except Exception as e:
        print(f"Get file exist time error: {str(e)}")
        return 99999999
    return (current_time - creation_time) / 60.0


def get_current_save_idx():
    current_time = time.time()
    save_idx = int(round(current_time, 4) * 10000)
    return save_idx


def get_step_cnt(answer):
    counter = 1
    if ' ки\n' in answer:
        counter = answer.count(' ки\n')
    return counter


def to_raw_string(string):
    encoded_string = string.encode('unicode_escape')
    raw_string = encoded_string.decode('ascii')
    return raw_string


def check_question_finished(question_path, question_parallel_num, num_sequence):
    if not is_file_exists(question_path):
        return False, 0
    if question_parallel_num == 0:
        question_parallel_num = 1

    file_list = [file for file in os.listdir(question_path) if file.endswith(".jsonl")]
    file_cnt = 0
    for file_name in file_list:
        file_path = os.path.join(question_path, file_name)
        try:
            with jsonlines.open(file_path, "r") as reader:
                for obj in reader:
                    output_temp = [obj['output'][k]['text'] for k in range(len(obj['output']))]
                    value_temp = [obj['output'][k]['value'] for k in range(len(obj['output']))]
                    if len(output_temp) == num_sequence:
                        file_cnt += 1
        except Exception as e:
            print(f"Read file error: {str(e)}")

    if file_cnt >= question_parallel_num:
        return True, file_cnt
    else:
        return False, file_cnt


def check_lock_timeout(raw_test_ds, question_parallel_num, save_dir, lock_dir, max_exist_time):
    if question_parallel_num == 0:
        question_parallel_num = 1

    for i in range(len(raw_test_ds)):
        for j in range(question_parallel_num):
            file_path = os.path.join(save_dir, f"question_{i}/record_{j}.jsonl")
            lock_file_path = os.path.join(save_dir, f"{lock_dir}/question_{i}_{j}.lock")
            if is_file_exists(lock_file_path):
                exist_time = get_file_exist_time(lock_file_path)
                if exist_time > max_exist_time or is_file_exists(file_path):
                    try:
                        os.remove(lock_file_path)
                    except Exception as e:
                        print(f"Remove lock file error: {str(e)}")


def check_process_cnt(raw_test_ds, question_parallel_num, save_dir):
    total_cnt = 0
    for i in range(len(raw_test_ds)):
        file_cnt = 0
        question_path = os.path.join(save_dir, f"question_{i}")
        if not is_file_exists(question_path):
            continue
        file_list = [file for file in os.listdir(question_path) if file.endswith(".jsonl")]
        for file_name in file_list:
            if not is_file_empty(os.path.join(question_path, file_name)):
                file_cnt += 1
            if file_cnt >= question_parallel_num:
                break
        total_cnt += file_cnt

    return total_cnt


def heart_beat_worker(chosen_idxs, save_dir, lock_dir):
    while True:
        for (i, j) in chosen_idxs:
            file_path = os.path.join(save_dir, f"question_{i}/record_{j}.jsonl")
            lock_file_path = os.path.join(save_dir, f"{lock_dir}/question_{i}_{j}.lock")
            if is_file_exists(lock_file_path):
                try:
                    os.utime(lock_file_path)
                except Exception as e:
                    print(f"Update lock file error: {str(e)}")
            else:
                try:
                    create_empty_file(lock_file_path)
                except Exception as e:
                    print(f"Create lock file error: {str(e)}")
        time.sleep(60)


def preprocess_data(problem_inst, file_path):
    data = copy.deepcopy(problem_inst)
    data['file_path'] = file_path
    return data


def assign_tasks(
    raw_test_ds, question_parallel_num, num_sequence, save_dir, lock_dir, batch_size=0, max_exist_time=0
):
    """Assign problems for current run."""

    check_lock_timeout(raw_test_ds, question_parallel_num, save_dir, lock_dir, max_exist_time)

    file_cnts = [0 for _ in range(len(raw_test_ds))]
    for i in range(len(raw_test_ds)):
        flag, file_cnt = check_question_finished(os.path.join(save_dir, f"question_{i}"), question_parallel_num, num_sequence)
        file_cnts[i] = file_cnt

    print(f'Batch size: {batch_size}, Max exist time: {max_exist_time}')

    start_time = time.time()
    print(f'Assign start at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')

    test_ds = []
    chosen_idxs = []
    chosen_dict = {}
    if question_parallel_num == 0:
        question_parallel_num = 1

    for j in range(question_parallel_num):
        for i in range(len(raw_test_ds)):
            file_path = os.path.join(save_dir, f"question_{i}/record_{j}.jsonl")
            lock_file_path = os.path.join(save_dir, f"{lock_dir}/question_{i}_{j}.lock")

            if not is_file_exists(file_path):
                if is_file_exists(lock_file_path):
                    exist_time = get_file_exist_time(lock_file_path)
                    if exist_time > max_exist_time:
                        try:
                            print(f"Remove lock file {lock_file_path}, exist time: {exist_time:.1f} minutes")
                            os.remove(lock_file_path)
                            chosen_idxs.append([i, j])
                            if i not in chosen_dict.keys():
                                chosen_dict[i] = [j]
                            else:
                                chosen_dict[i].append(j)
                            test_ds.append(preprocess_data(raw_test_ds[i], file_path))
                            create_empty_file(lock_file_path)
                        except Exception as e:
                            print(f"Remove lock file error: {str(e)}")
                            continue
                    else:
                        continue
                else:
                    chosen_idxs.append([i, j])
                    if i not in chosen_dict.keys():
                        chosen_dict[i] = [j]
                    else:
                        chosen_dict[i].append(j)
                    test_ds.append(preprocess_data(raw_test_ds[i], file_path))
                    create_empty_file(lock_file_path)
            if len(chosen_idxs) >= batch_size:
                break
        if len(chosen_idxs) >= batch_size:
            break

    print(f"Len: {len(test_ds)}, Chosen dict: {chosen_dict}")
    print(f"Assign end at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}, Time cost: {time.time() - start_time:.1f} seconds")

    refresh_thread = threading.Thread(target=heart_beat_worker, args=(chosen_idxs, save_dir, lock_dir), daemon=True)
    refresh_thread.start()

    return test_ds, chosen_dict


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank(message):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print("[{}/{}]: {}".format(rank, world_size, message), flush=True)
    else:
        print(message, flush=True)
