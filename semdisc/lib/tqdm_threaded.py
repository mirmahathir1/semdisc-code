import threading
import os
from tqdm import tqdm
import time
from custom_lib import utils, console
class TqdmFileSubmitter():
    def __init__(self, total_combination_count, multiplier, tracker_path):
        self.total_combination_count = total_combination_count
        self.multiplier = multiplier
        self.tracker_path = tracker_path
        self.counter = 0
    def check_and_submit_tracker_file(self):
        self.counter += 1
        if self.counter % self.multiplier == 0:
            TqdmThreaded.submit_join_graph_tracker_file(tracker_path=self.tracker_path, filename_string=utils.get_random_hash())
        
class TqdmThreaded(threading.Thread):
    def __init__(self, total_combination_count, tracker_path, multiplier, percentage_of_progress_for_update, process_count, delete_files = True):
        super(TqdmThreaded, self).__init__()
        self.delete_files = delete_files
        self.running = True
        self.total_combination_count = total_combination_count
        self.pbar = tqdm(total=self.total_combination_count)
        
        self.multiplier = multiplier
        if self.multiplier == 1 and self.total_combination_count is not None and percentage_of_progress_for_update is not None:
            self.multiplier = min(max(int(self.total_combination_count * percentage_of_progress_for_update), 1), max(int(self.total_combination_count / process_count), 1))

    def produce_tracker_file_submitter(self):
        return TqdmFileSubmitter(total_combination_count = self.total_combination_count, multiplier = self.multiplier, tracker_path = self.tracker_path)

    def run(self):
        self.total_completed = 0
        while self.running:
            tracker_files = os.listdir(self.tracker_path)

            if self.delete_files:
                current_completed = len(tracker_files) * self.multiplier
            else:
                current_completed = len(tracker_files) * self.multiplier - self.total_completed

            # current_completed = len(tracker_files) * self.multiplier - self.total_completed
            if current_completed > 0:
                self.pbar.update(current_completed)
                self.total_completed += current_completed
                console.log(f"TQDM {self.total_completed}/{self.total_combination_count}", standard_output=False)
            if self.delete_files:
                for file in tracker_files:
                    os.remove(f"{self.tracker_path}/{file}")
            time.sleep(1)

    def stop(self):
        if not self.total_combination_count is None:
            self.pbar.update(self.total_combination_count - self.total_completed)
            console.log(f"TQDM {self.total_combination_count}/{self.total_combination_count}", standard_output=False)
        self.running = False
        self.pbar.close()

    @staticmethod
    def submit_join_graph_tracker_file(tracker_path, filename_string):
        open(f"{tracker_path}/{filename_string}.txt", 'w')

class TqdmNonThreaded(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completed = 0
    def update(self, n=1):
        super().update(n)
        self.completed += n
        console.log(f"TQDM {self.completed}/{self.total}", standard_output=False)

    def close(self):
        super().close()
        if not self.total is None:
            console.log(f"TQDM {self.total}/{self.total}", standard_output=False)
