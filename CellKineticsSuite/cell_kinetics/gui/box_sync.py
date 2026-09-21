"""Background worker that watches a Box-synced folder and copies complete
dataset triplets into the local working folder as they arrive.

Built on core.dataset_io so the same prefix/channel-matching logic is used
here and in the local-folder loader -- previously these were two separately
maintained, near-identical blocks of regex/matching code.
"""
import os
import shutil
import threading
import time

from ..core import dataset_io


class BoxSyncWorker:
    """dataset_prefixes and sync_lock are shared with the caller (typically
    the main window) and mutated in place, so both sides always see the
    same queue -- mirroring how this worked before extraction, just with
    the polling logic now living in its own class.

    on_dataset_synced() is dispatched (main thread) whenever a prefix is
    newly added or its files were updated. on_status_changed() is dispatched
    once per poll cycle so counters/pending-reasons stay fresh even when
    nothing changed.
    """

    def __init__(self, dispatcher, dataset_prefixes, sync_lock, on_dataset_synced, on_status_changed):
        self.dispatcher = dispatcher
        self.dataset_prefixes = dataset_prefixes
        self.sync_lock = sync_lock
        self.on_dataset_synced = on_dataset_synced
        self.on_status_changed = on_status_changed

        self.box_source_path = None
        self.master_folder_path = None
        self.pending_reasons = {}
        self.last_sync_log = "Idle"
        self.count_downloaded = 0
        self.count_pending = 0

        self.active = False
        self._thread = None
        self._poll_interval_sec = 3.0

    def start(self, box_source_path, master_folder_path):
        self.box_source_path = box_source_path
        self.master_folder_path = master_folder_path
        self.last_sync_log = f"Listening on: {os.path.basename(box_source_path)}"
        self.active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False

    def _run(self):
        while self.active:
            try:
                self._poll_once()
            except Exception as e:
                self.last_sync_log = f"Sync error: {str(e)[:25]}"
            time.sleep(self._poll_interval_sec)

    def _poll_once(self):
        if not (os.path.exists(self.box_source_path) and os.path.exists(self.master_folder_path)):
            return

        box_files = os.listdir(self.box_source_path)
        all_box_prefixes = {
            dataset_io.split_prefix(f) for f in box_files if f.lower().endswith(".txt")
        }

        pending_count = 0
        for prefix in all_box_prefixes:
            channel_files = dataset_io.find_channel_files(prefix, box_files)

            if not dataset_io.is_complete(channel_files):
                pending_count += 1
                missing = dataset_io.missing_channels(channel_files)
                self.pending_reasons[prefix] = f"missing {', '.join(missing)} file"
                continue

            files_to_copy = [channel_files["gfp"], channel_files["mcherry"], channel_files["mask"]]
            if channel_files["param"]:
                files_to_copy.append(channel_files["param"])

            all_ready = all(
                os.path.getsize(os.path.join(self.box_source_path, f)) > 0 for f in files_to_copy
            )
            if not all_ready:
                pending_count += 1
                self.pending_reasons[prefix] = "files found but still 0 bytes (mid-upload?)"
                continue

            copied_any = False
            for f in files_to_copy:
                src = os.path.join(self.box_source_path, f)
                dst = os.path.join(self.master_folder_path, f)
                if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                    shutil.copy2(src, dst)
                    copied_any = True

            with self.sync_lock:
                self.pending_reasons.pop(prefix, None)
                if prefix not in self.dataset_prefixes:
                    self.dataset_prefixes.append(prefix)
                    self.dataset_prefixes.sort()
                    self.last_sync_log = f"Synced: {prefix}"
                    self.dispatcher.post(self.on_dataset_synced)
                elif copied_any:
                    self.last_sync_log = f"Updated files for: {prefix}"
                    self.dispatcher.post(self.on_dataset_synced)

        self.count_downloaded = len(self.dataset_prefixes)
        self.count_pending = pending_count
        self.dispatcher.post(self.on_status_changed)
