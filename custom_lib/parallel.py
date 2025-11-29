from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import time
from custom_lib import utils, console
import threading
import numpy as np
import sys
import signal
import traceback
import random
class TqdmFileSubmitter():
    def __init__(self, total_combination_count, multiplier):
        self.total_combination_count = total_combination_count
        self.multiplier = multiplier
        self.counter = 0

    def check_and_submit_to_common_counter(self, lock, common_counter):
        self.counter += 1
        if self.counter % self.multiplier == 0:
            with lock:
                common_counter.value += self.multiplier
        
class TqdmThreaded(threading.Thread):
    def __init__(self,
                 total_combination_count,
                 percentage_of_progress_for_update,
                 process_count,
                 counter,
                 lock):
        super(TqdmThreaded, self).__init__()
        self.process_count = process_count
        self.running = True
        self.total_combination_count = total_combination_count

        per_process_item_count = max(int(self.total_combination_count / process_count),1)
        count_of_finished_tasks_needed_for_tqdm_update = int(self.total_combination_count * percentage_of_progress_for_update)
        self.multiplier = max(min(per_process_item_count, count_of_finished_tasks_needed_for_tqdm_update),1)
        console.log(f"tqdm info: process count = {self.process_count}, multiplier of update = {self.multiplier}")

        self.counter = counter
        self.lock = lock
        self.pbar = tqdm(total=self.total_combination_count)

    def produce_tracker_submitter(self):
        return TqdmFileSubmitter(total_combination_count = self.total_combination_count, multiplier = self.multiplier)

    def run(self):
        self.total_completed = 0
        while self.running:
            current_completed = self.counter.value - self.total_completed
            if current_completed > 0:
                self.pbar.update(current_completed)
                self.total_completed += current_completed
            time.sleep(1)

    def kill(self):
        self.running = False
        self.pbar.close()

    def stop(self):
        if not self.total_combination_count is None:
            self.pbar.update(self.total_combination_count - self.total_completed)
        self.running = False
        self.pbar.close()

def divide_list(lst, n):
    length = len(lst)
    size = length // n
    remainder = length % n
    start = 0
    result = []
    for i in range(n):
        end = start + size
        if remainder:
            end += 1
            remainder -= 1
        if len(lst[start:end]) == 0:
            continue
        result.append(lst[start:end])
        start = end
    return result

# function for one process to run
def batch_task(func, argument_list, common_arguments, tracker_submitter,
               counter, lock, function_for_batch
               ):
    results = []
    variable_for_batch = function_for_batch()
    for argument in argument_list:
        result = func(argument, common_arguments, variable_for_batch)
        results.append(result)
        tracker_submitter.check_and_submit_to_common_counter(lock, counter)
    return results

def init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# function to distribute work to multiple processes
def execute(func, argument_list, common_arguments={}, function_for_batch=None, parallel_enabled=True):
    console.log(str(func), f"parallel enabled: {parallel_enabled}")
    if function_for_batch is None:
        function_for_batch = utils.empty_function

    cmd_arguments = utils.parse_arg([
        {'name': 'pcount', 'type': int, 'default': 32},
        {'name': 'parallel', 'type': str, 'default': "True"}
    ])

    if parallel_enabled == True:
        parallel_enabled = cmd_arguments['parallel'] == "True"

    # --- serial path ------------------------------------------------------
    if not parallel_enabled:
        results = []
        scratch = function_for_batch()
        for arg in tqdm(argument_list):
            results.append(func(arg, common_arguments, scratch))
        return results
    # ---------------------------------------------------------------------



    random.shuffle(argument_list)

    process_count = cmd_arguments['pcount']
    # process_count = 32
    percentage_of_progress_for_tqdm = 0.01
    assert callable(func), "func must be a function"
    assert isinstance(argument_list, list), "argument_list must be a list"
    assert isinstance(common_arguments, dict), "common_arguments must be a dictionary"
    assert isinstance(process_count, int), "process_count must be an integer"
    assert isinstance(percentage_of_progress_for_tqdm, float), "percentage_of_progress_for_tqdm must be a float"
    if utils.is_function_running_under_flask():
        process_count = 1

    slurm_cpu_count = utils.get_avaiable_cpu_core_for_slurm_job()
    if slurm_cpu_count is None:
        slurm_cpu_count = 99999999
    else:
        slurm_cpu_count = int(slurm_cpu_count)
    process_count = np.min([slurm_cpu_count, cpu_count(), process_count])

    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    parallel_argument_lists = divide_list(argument_list, process_count)
    with Pool(processes=process_count, initializer=init) as p:
        tracker = TqdmThreaded(total_combination_count=len(argument_list),
                               percentage_of_progress_for_update=percentage_of_progress_for_tqdm,
                               process_count=process_count,
                               counter=counter,
                               lock=lock,
                               )
        tracker.start()
        try:
            starmap_result_list = p.starmap(batch_task, [(func,
                                                        argument_list,
                                                        common_arguments,
                                                        tracker.produce_tracker_submitter(),
                                                        counter,
                                                        lock,
                                                        function_for_batch
                                                        ) for argument_list in parallel_argument_lists])
        except Exception as e:
            console.log("Error:", e)
            traceback.print_exc()
            p.terminate()
            p.join()
            tracker.kill()
            tracker.join()
            sys.exit(1)

        tracker.stop()
        tracker.join()
    all_result = []
    for starmap_result in starmap_result_list:
        all_result.extend(starmap_result)
    return all_result

"""

parallel.execute(func, argument_list, common_arguments) expects 
(i) a worker function func(arg, common_args, scratch) that processes a single item, 
(ii) an argument_list of items to feed that function, 
(iii) an optional common_arguments dictionary that is passed unchanged to every call

Each call to your worker function receives **three positional inputs**: 
(1) `argument` – the individual element taken from `argument_list`, representing the specific task instance the worker should process; 
(2) `common_arguments` – the same read-only dictionary for every call, useful for configuration knobs, 
constants, or shared resources that do **not** need to be recreated each time; and 
(3) `common_arguments_for_batch`- ignore the details of this.


After distributing the work in parallel, execute returns one flat list containing the result produced by func for every item in argument_list; 
the ordering is arbitrary because it reflects how the chunks complete.
"""