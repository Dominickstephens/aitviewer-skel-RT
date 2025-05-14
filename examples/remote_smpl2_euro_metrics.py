import sys
import time
import numpy as np
import socket
import pickle
import zlib
import struct
import threading
import os
import traceback # For detailed error printing
import math # Needed for math.pi in One Euro Filter
import csv # For CSV writing
import tkinter as tk # For GUI
from tkinter import ttk # For GUI dropdown
from collections import deque # For metric buffering
import queue # For passing data to metric thread

# --- AITViewer Imports ---
try:
    from aitviewer.remote.renderables.smpl import RemoteSMPLSequence
    from aitviewer.remote.viewer import RemoteViewer
except ImportError as e:
    print(f"ERROR: Failed to import aitviewer components: {e}")
    print("Ensure aitviewer is installed correctly in your environment.")
    sys.exit(1)


# --- Filter Implementations ---

# --- One Euro Filter (NumPy compatible) ---
def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev): # This is essentially Single Exponential Smoothing
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    def __init__(self, x0, t0=None, dx0=None, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        if not isinstance(x0, np.ndarray):
            raise ValueError("Initial value x0 must be a numpy array")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.data_shape = x0.shape
        self.x_prev = x0.astype(np.float64)
        if dx0 is None:
            self.dx_prev = np.zeros_like(x0, dtype=np.float64)
        elif isinstance(dx0, np.ndarray) and dx0.shape == x0.shape:
            self.dx_prev = dx0.astype(np.float64)
        else:
            raise ValueError("dx0 must be None or a numpy array of the same shape as x0")
        self.t_prev = float(t0) if t0 is not None else time.time()
        # Store parameters for comparison during re-initialization
        self.params = {"min_cutoff": self.min_cutoff, "beta": self.beta, "d_cutoff": self.d_cutoff, "type": "one_euro"}


    def __call__(self, t, x):
        if not isinstance(x, np.ndarray):
            raise ValueError("Input value 'x' must be a numpy array")
        if x.shape != self.data_shape: 
            print(f"Warning: OneEuroFilter input shape mismatch. Expected {self.data_shape}, got {x.shape}. Resetting filter with new shape.")
            self.data_shape = x.shape
            self.x_prev = x.astype(np.float64) 
            self.dx_prev = np.zeros_like(x, dtype=np.float64) 
            self.t_prev = float(t) 
            return x.copy() 
            
        x_array = x.astype(np.float64)
        t_scalar = float(t)
        t_e = t_scalar - self.t_prev

        if t_e <= 1e-9: 
            return self.x_prev.copy() 

        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x_array - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)
        speed_magnitude = np.linalg.norm(dx_hat) 
        cutoff = self.min_cutoff + self.beta * speed_magnitude
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x_array, self.x_prev)
        
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t_scalar
        return x_hat.copy()

# --- Single Exponential Smoothing Filter ---
class SingleExponentialFilter:
    def __init__(self, x0, t0=None, alpha=0.5): # t0 is for interface consistency, not used by basic SES
        if not isinstance(x0, np.ndarray):
            raise ValueError("Initial value x0 must be a numpy array for SingleExponentialFilter")
        self.alpha = float(alpha)
        self.x_prev = x0.astype(np.float64)
        self.data_shape = x0.shape
        self.params = {"alpha": self.alpha, "type": "single_exp"}


    def __call__(self, t, x): # t is for interface consistency
        if not isinstance(x, np.ndarray) or x.shape != self.data_shape:
            print(f"Warning: SingleExponentialFilter shape mismatch or wrong type. Resetting. Expected {self.data_shape}, got {x.shape if isinstance(x,np.ndarray) else type(x)}")
            self.data_shape = x.shape if isinstance(x, np.ndarray) else self.data_shape
            self.x_prev = x.astype(np.float64) if isinstance(x, np.ndarray) else self.x_prev
            return x.copy() if isinstance(x, np.ndarray) else self.x_prev.copy()

        x_array = x.astype(np.float64)
        self.x_prev = self.alpha * x_array + (1 - self.alpha) * self.x_prev
        return self.x_prev.copy()

# --- Double Exponential Smoothing Filter (Holt's Linear Trend) ---
class DoubleExponentialFilter:
    def __init__(self, x0, t0=None, alpha=0.5, beta_trend=0.5): # t0 for consistency
        if not isinstance(x0, np.ndarray):
            raise ValueError("Initial value x0 must be a numpy array for DoubleExponentialFilter")
        self.alpha = float(alpha)
        self.beta_trend = float(beta_trend)
        self.level_prev = x0.astype(np.float64) # Initial level is the first observation
        self.trend_prev = np.zeros_like(x0, dtype=np.float64) # Initial trend can be zero or (x1-x0)
        self.data_shape = x0.shape
        self.params = {"alpha": self.alpha, "beta_trend": self.beta_trend, "type": "double_exp"}


    def __call__(self, t, x): # t for consistency
        if not isinstance(x, np.ndarray) or x.shape != self.data_shape:
            print(f"Warning: DoubleExponentialFilter shape mismatch or wrong type. Resetting. Expected {self.data_shape}, got {x.shape if isinstance(x,np.ndarray) else type(x)}")
            self.data_shape = x.shape if isinstance(x, np.ndarray) else self.data_shape
            self.level_prev = x.astype(np.float64) if isinstance(x, np.ndarray) else self.level_prev
            self.trend_prev = np.zeros_like(self.level_prev, dtype=np.float64)
            return x.copy() if isinstance(x, np.ndarray) else self.level_prev.copy()

        x_array = x.astype(np.float64)
        
        current_level = self.alpha * x_array + (1 - self.alpha) * (self.level_prev + self.trend_prev)
        current_trend = self.beta_trend * (current_level - self.level_prev) + (1 - self.beta_trend) * self.trend_prev
        
        self.level_prev = current_level
        self.trend_prev = current_trend
        
        # The filtered value is the current level (some implementations use level + trend for forecast)
        # For smoothing current observation, current_level is appropriate.
        return self.level_prev.copy()

# --- Moving Average Filter ---
class MovingAverageFilter:
    def __init__(self, x0, t0=None, window_size=5): # t0 for consistency
        self.window_size = int(window_size)
        if self.window_size < 1: self.window_size = 1 # Ensure window size is at least 1
        
        if not isinstance(x0, np.ndarray):
             # This case should ideally not happen if x0 is always an ndarray
            raise ValueError("Initial value x0 must be a numpy array for MovingAverageFilter")

        self.buffer = deque([x0.astype(np.float64)] * self.window_size, maxlen=self.window_size)
        self.data_shape = x0.shape # Store initial shape
        self.params = {"window_size": self.window_size, "type": "moving_avg"}


    def __call__(self, t, x): # t for consistency
        if not isinstance(x, np.ndarray) or x.shape != self.data_shape:
            print(f"Warning: MovingAverageFilter shape mismatch or wrong type. Resetting. Expected {self.data_shape}, got {x.shape if isinstance(x,np.ndarray) else type(x)}")
            self.data_shape = x.shape if isinstance(x, np.ndarray) else self.data_shape
            if isinstance(x, np.ndarray):
                self.buffer = deque([x.astype(np.float64)] * self.window_size, maxlen=self.window_size)
                return x.copy()
            else: 
                return np.mean(np.array(list(self.buffer)), axis=0) if self.buffer else np.zeros(self.data_shape)

        x_array = x.astype(np.float64)
        self.buffer.append(x_array)
        return np.mean(np.array(list(self.buffer)), axis=0)

# --- End Filter Implementations ---

# --- Configuration ---
VIEWER_HOST = "localhost"; VIEWER_PORT = 8417
RELAY_LISTEN_HOST = 'localhost'; RELAY_LISTEN_PORT = 9999
NUM_BETAS_CONFIG = 10; POSE_BODY_DIM_CONFIG = 69; MAGIC_HEADER = b'SMPL'

# --- Filter Presets ---
FILTER_PRESETS = {
    "No Filter Active": {
        "type": "none", 
        "poses_body": {"enabled": False}, "poses_root": {"enabled": False},
        "betas": {"enabled": False}, "trans": {"enabled": False}
    },
    "OneEuro Moderate": { 
        "type": "one_euro",
        "poses_body": {"min_cutoff": 1.0, "beta": 0.5, "d_cutoff": 1.0, "enabled": True},
        "poses_root": {"min_cutoff": 1.0, "beta": 0.5, "d_cutoff": 1.0, "enabled": True},
        "betas": {"min_cutoff": 0.5, "beta": 0.2, "d_cutoff": 1.0, "enabled": True},
        "trans": {"min_cutoff": 1.0, "beta": 0.7, "d_cutoff": 1.0, "enabled": True},
    },
    "OneEuro Smooth": {
        "type": "one_euro",
        "poses_body": {"min_cutoff": 0.3, "beta": 0.05, "d_cutoff": 0.8, "enabled": True},
        "poses_root": {"min_cutoff": 0.3, "beta": 0.05, "d_cutoff": 0.8, "enabled": True},
        "betas": {"min_cutoff": 0.1, "beta": 0.01, "d_cutoff": 0.5, "enabled": True},
        "trans": {"min_cutoff": 0.4, "beta": 0.1, "d_cutoff": 0.9, "enabled": True},
    },
    "Single Exp (Alpha 0.3 Smooth)": { # Renamed for clarity
        "type": "single_exp",
        "common_params": {"alpha": 0.3}, 
        "poses_body": {"enabled": True}, "poses_root": {"enabled": True},
        "betas": {"enabled": True}, "trans": {"enabled": True}
    },
    "Single Exp (Alpha 0.7 Responsive)": { # New Variant
        "type": "single_exp",
        "common_params": {"alpha": 0.7}, 
        "poses_body": {"enabled": True}, "poses_root": {"enabled": True},
        "betas": {"enabled": True}, "trans": {"enabled": True}
    },
    "Double Exp (A0.3 B0.1 Smooth)": { # Renamed for clarity
        "type": "double_exp",
        "common_params": {"alpha": 0.3, "beta_trend": 0.1}, 
        "poses_body": {"enabled": True}, "poses_root": {"enabled": True},
        "betas": {"enabled": True}, "trans": {"enabled": True}
    },
    "Double Exp (A0.7 B0.3 Responsive)": { # New Variant
        "type": "double_exp",
        "common_params": {"alpha": 0.7, "beta_trend": 0.3}, 
        "poses_body": {"enabled": True}, "poses_root": {"enabled": True},
        "betas": {"enabled": True}, "trans": {"enabled": True}
    },
    "Moving Avg (Win 3 Responsive)": { # New Variant (more responsive)
        "type": "moving_avg",
        "common_params": {"window_size": 3}, 
        "poses_body": {"enabled": True}, "poses_root": {"enabled": True},
        "betas": {"enabled": True}, "trans": {"enabled": True}
    },
    "Moving Avg (Win 5 Moderate)": { # Renamed for clarity
        "type": "moving_avg",
        "common_params": {"window_size": 5}, 
        "poses_body": {"enabled": True}, "poses_root": {"enabled": True},
        "betas": {"enabled": True}, "trans": {"enabled": True}
    },
    "Moving Avg (Win 10 Smooth)": { # New Variant (smoother)
        "type": "moving_avg",
        "common_params": {"window_size": 10}, 
        "poses_body": {"enabled": True}, "poses_root": {"enabled": True},
        "betas": {"enabled": True}, "trans": {"enabled": True}
    }
}
# Initialize active filter settings to one of the presets
active_filter_settings = FILTER_PRESETS["OneEuro Moderate"].copy() 
filter_settings_lock = threading.Lock() 

# --- Metrics Configuration & Globals ---
METRIC_BUFFER_DURATION_SECONDS_RELAY = 5
MAX_BUFFER_LEN_RELAY = int(60 * (METRIC_BUFFER_DURATION_SECONDS_RELAY + 3)) 

raw_pose_body_change_buffer = deque(maxlen=MAX_BUFFER_LEN_RELAY)
raw_pose_root_change_buffer = deque(maxlen=MAX_BUFFER_LEN_RELAY)
raw_trans_change_buffer = deque(maxlen=MAX_BUFFER_LEN_RELAY)
raw_betas_variance_buffer = deque(maxlen=MAX_BUFFER_LEN_RELAY)
filtered_pose_body_change_buffer = deque(maxlen=MAX_BUFFER_LEN_RELAY)
filtered_pose_root_change_buffer = deque(maxlen=MAX_BUFFER_LEN_RELAY)
filtered_trans_change_buffer = deque(maxlen=MAX_BUFFER_LEN_RELAY)
filtered_betas_variance_buffer = deque(maxlen=MAX_BUFFER_LEN_RELAY)
filter_latency_buffer = deque(maxlen=MAX_BUFFER_LEN_RELAY)
relay_packet_processing_fps_buffer = deque(maxlen=MAX_BUFFER_LEN_RELAY)

CSV_FILENAME_RELAY = "relay_filter_metrics_avg.csv"
CSV_FIELDNAMES_RELAY = [
    "Timestamp", "Condition", "FilterSetting", 
    "Avg Relay Packet Processing FPS", "Avg Filter Latency (ms)",
    "Avg Raw Pose Body Change", "Avg Filtered Pose Body Change",
    "Avg Raw Pose Root Change", "Avg Filtered Pose Root Change",
    "Avg Raw Trans Change (mm)", "Avg Filtered Trans Change (mm)",
    "Avg Raw Betas Variance", "Avg Filtered Betas Variance"
]

latest_smpl_data_for_viewer = None 
data_lock_viewer = threading.Lock() 
client_has_connected = threading.Event() 
stop_all_threads = threading.Event() 
gui_root_relay = None; condition_var_relay = None 
relay_metric_data_queue = queue.Queue(maxsize=30) 
relay_metric_thread_instance = None

# --- Socket Utilities ---
def recv_n_bytes(sock, n):
    buf = b"";
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: raise ConnectionError(f"Socket closed with {n - len(buf)} bytes left")
        buf += chunk
    return buf

def recv_message(conn):
    hdr = recv_n_bytes(conn, 4 + 4 + 4) 
    magic, length, sent_crc = struct.unpack('>4s I I', hdr)
    if magic != MAGIC_HEADER: raise ValueError(f"Bad magic: {magic!r}")
    payload = recv_n_bytes(conn, length)
    if zlib.crc32(payload) & 0xFFFFFFFF != sent_crc: raise ValueError("Checksum mismatch")
    return pickle.loads(payload)

# --- Metric Calculation Thread ---
def metrics_calculation_worker_relay():
    global stop_all_threads, relay_metric_data_queue
    global raw_pose_body_change_buffer, raw_pose_root_change_buffer, raw_trans_change_buffer, raw_betas_variance_buffer
    global filtered_pose_body_change_buffer, filtered_pose_root_change_buffer, filtered_trans_change_buffer, filtered_betas_variance_buffer
    global filter_latency_buffer, relay_packet_processing_fps_buffer

    print("Relay metrics calculation thread started.")
    while not stop_all_threads.is_set():
        try:
            data_packet = relay_metric_data_queue.get(timeout=0.1) 
            if data_packet is None: break 

            ts = data_packet["timestamp"]
            raw_data = data_packet["raw_data"] 
            filt_data = data_packet["filtered_data"] 
            prev_raw = data_packet["prev_raw_data_for_comp"] 
            prev_filt = data_packet["prev_filtered_data_for_comp"] 
            filter_lat = data_packet["filter_latency_ms"]
            relay_proc_fps = data_packet["relay_packet_processing_fps"]

            raw_pb_change, raw_pr_change, raw_tr_change, raw_be_var = np.nan, np.nan, np.nan, np.nan
            filt_pb_change, filt_pr_change, filt_tr_change, filt_be_var = np.nan, np.nan, np.nan, np.nan

            if raw_data and prev_raw and prev_raw.get("valid_for_comparison"):
                if "poses_body" in raw_data and "poses_body" in prev_raw:
                    diff_pb_raw = raw_data["poses_body"] - prev_raw["poses_body"]
                    raw_pb_change = np.linalg.norm(diff_pb_raw) 
                if "poses_root" in raw_data and "poses_root" in prev_raw:
                    diff_pr_raw = raw_data["poses_root"] - prev_raw["poses_root"]
                    raw_pr_change = np.linalg.norm(diff_pr_raw)
                if "trans" in raw_data and "trans" in prev_raw: 
                    raw_tr_change = np.linalg.norm((raw_data["trans"] * 1000) - (prev_raw["trans"] * 1000))
            if raw_data and "betas" in raw_data:
                raw_be_var = np.var(raw_data["betas"])

            if filt_data and prev_filt and prev_filt.get("valid_for_comparison"):
                if "poses_body" in filt_data and "poses_body" in prev_filt:
                    diff_pb_filt = filt_data["poses_body"] - prev_filt["poses_body"]
                    filt_pb_change = np.linalg.norm(diff_pb_filt)
                if "poses_root" in filt_data and "poses_root" in prev_filt:
                    diff_pr_filt = filt_data["poses_root"] - prev_filt["poses_root"]
                    filt_pr_change = np.linalg.norm(diff_pr_filt)
                if "trans" in filt_data and "trans" in prev_filt: 
                    filt_tr_change = np.linalg.norm((filt_data["trans"] * 1000) - (prev_filt["trans"] * 1000))
            if filt_data and "betas" in filt_data:
                filt_be_var = np.var(filt_data["betas"])

            raw_pose_body_change_buffer.append((ts, raw_pb_change))
            raw_pose_root_change_buffer.append((ts, raw_pr_change))
            raw_trans_change_buffer.append((ts, raw_tr_change))
            raw_betas_variance_buffer.append((ts, raw_be_var))
            filtered_pose_body_change_buffer.append((ts, filt_pb_change))
            filtered_pose_root_change_buffer.append((ts, filt_pr_change))
            filtered_trans_change_buffer.append((ts, filt_tr_change))
            filtered_betas_variance_buffer.append((ts, filt_be_var))
            filter_latency_buffer.append((ts, filter_lat))
            relay_packet_processing_fps_buffer.append((ts, relay_proc_fps))

            relay_metric_data_queue.task_done() 
        except queue.Empty: 
            continue 
        except Exception as e:
            print(f"Error in relay_metrics_calculation_worker: {e}"); traceback.print_exc()
    print("Relay metrics calculation thread finished.")

# --- GUI Setup and Callbacks ---
def on_filter_condition_change(*args): 
    global active_filter_settings, filter_settings_lock, condition_var_relay, FILTER_PRESETS
    selected_condition_name = condition_var_relay.get()
    print(f"Relay GUI: Filter condition changed to: {selected_condition_name}")
    if selected_condition_name in FILTER_PRESETS:
        new_preset = FILTER_PRESETS[selected_condition_name]
        with filter_settings_lock: 
            # Deep copy preset to avoid modifying the original FILTER_PRESETS dict
            active_filter_settings = {key: val.copy() if isinstance(val, dict) else val for key, val in new_preset.items()}
            if 'type' not in active_filter_settings and 'type' in new_preset: # Ensure top-level type is copied
                active_filter_settings['type'] = new_preset['type']
            print(f"Relay: Active filter settings updated to type: '{active_filter_settings.get('type', 'N/A')}' (Preset: '{selected_condition_name}')")
    else:
        print(f"Relay Warning: Selected condition '{selected_condition_name}' not found. Filters unchanged.")

def setup_gui_relay():
    global gui_root_relay, condition_var_relay, stop_all_threads, active_filter_settings 
    gui_root_relay = tk.Tk(); gui_root_relay.title("Relay Metrics & Filter Control"); gui_root_relay.geometry("400x200") 
    def on_gui_close_relay(): global stop_all_threads; print("Relay GUI closed."); stop_all_threads.set(); gui_root_relay.destroy()
    gui_root_relay.protocol("WM_DELETE_WINDOW", on_gui_close_relay)
    tk.Label(gui_root_relay, text="Filter Condition:").pack(pady=(10,0))
    
    filter_condition_names = list(FILTER_PRESETS.keys()) # Get names from presets
    condition_var_relay = tk.StringVar(gui_root_relay)
    
    # Determine initial GUI selection based on current active_filter_settings
    initial_gui_condition_name = "No Filter Active" # Fallback
    with filter_settings_lock:
        # This comparison needs to be robust if active_filter_settings can be modified beyond presets
        for name, preset_dict_iter in FILTER_PRESETS.items():
            if active_filter_settings == preset_dict_iter: # Simple dict comparison
                initial_gui_condition_name = name
                break
    condition_var_relay.set(initial_gui_condition_name) 

    condition_dropdown_relay = ttk.OptionMenu(gui_root_relay, condition_var_relay, initial_gui_condition_name, *filter_condition_names)
    condition_dropdown_relay.pack(pady=5, padx=10, fill='x')
    condition_var_relay.trace_add("write", on_filter_condition_change) 

    tk.Button(gui_root_relay, text="Record Relay Avg Metrics (5s)", command=record_metrics_action_relay, height=2).pack(pady=10, padx=10, fill='x')
    status_label_var_relay = tk.StringVar(); status_label_var_relay.set("Select filter, then record.")
    tk.Label(gui_root_relay, textvariable=status_label_var_relay, wraplength=380).pack(pady=5)
    gui_root_relay.status_label_var = status_label_var_relay
    try: gui_root_relay.mainloop()
    except Exception as e: print(f"Relay GUI mainloop error: {e}")
    finally: 
        print("Relay GUI mainloop exited."); stop_all_threads.set()

def record_metrics_action_relay():
    global CSV_FILENAME_RELAY, CSV_FIELDNAMES_RELAY, condition_var_relay, gui_root_relay, METRIC_BUFFER_DURATION_SECONDS_RELAY, active_filter_settings
    global raw_pose_body_change_buffer, raw_pose_root_change_buffer, raw_trans_change_buffer, raw_betas_variance_buffer
    global filtered_pose_body_change_buffer, filtered_pose_root_change_buffer, filtered_trans_change_buffer, filtered_betas_variance_buffer
    global filter_latency_buffer, relay_packet_processing_fps_buffer

    if stop_all_threads.is_set(): 
        if gui_root_relay and hasattr(gui_root_relay,'status_label_var'): gui_root_relay.status_label_var.set("App shutting down.")
        return

    recording_time = time.time()
    gui_condition_label = condition_var_relay.get() 
    
    current_filter_name_for_log = "Unknown/Custom" 
    with filter_settings_lock: 
        # Try to find the name of the currently active filter preset
        # This is more robust if active_filter_settings is a deep copy or slightly modified
        active_type = active_filter_settings.get("type")
        active_common_params = active_filter_settings.get("common_params")

        for preset_name_iter, preset_values_iter in FILTER_PRESETS.items():
            if preset_values_iter.get("type") == active_type:
                if active_type == "one_euro": # OneEuro has per-stream settings
                    # For OneEuro, a direct dict comparison is more reliable if it's a direct copy
                    if active_filter_settings == preset_values_iter:
                        current_filter_name_for_log = preset_name_iter
                        break
                elif active_type == "none":
                     current_filter_name_for_log = preset_name_iter # Should be "No Filter Active"
                     break
                elif active_type in ["single_exp", "double_exp", "moving_avg"]:
                    # Compare common_params for these simpler filters
                    if preset_values_iter.get("common_params") == active_common_params:
                        current_filter_name_for_log = preset_name_iter
                        break
        if current_filter_name_for_log == "Unknown/Custom" and active_filter_settings.get("type") == "none":
             current_filter_name_for_log = "No Filter Active" # Ensure "No Filter Active" is logged correctly


    avg_metrics_log = {
        "Timestamp":time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(recording_time)), 
        "Condition":gui_condition_label, 
        "FilterSetting": current_filter_name_for_log 
    }

    def get_avg(d_buffer, nan_policy='omit'):
        buffer_list_snapshot = list(d_buffer) 
        relevant_metric_values = [val for ts, val in buffer_list_snapshot if ts >= recording_time - METRIC_BUFFER_DURATION_SECONDS_RELAY]
        if not relevant_metric_values: return np.nan
        if nan_policy == 'omit':
            valid_numeric_vals = [v for v in relevant_metric_values if not (isinstance(v,float) and np.isnan(v))]
            return np.mean(valid_numeric_vals) if valid_numeric_vals else np.nan
        else: 
            return np.mean(relevant_metric_values)

    avg_metrics_log["Avg Relay Packet Processing FPS"] = get_avg(relay_packet_processing_fps_buffer)
    avg_metrics_log["Avg Filter Latency (ms)"] = get_avg(filter_latency_buffer)
    avg_metrics_log["Avg Raw Pose Body Change"] = get_avg(raw_pose_body_change_buffer)
    avg_metrics_log["Avg Filtered Pose Body Change"] = get_avg(filtered_pose_body_change_buffer)
    avg_metrics_log["Avg Raw Pose Root Change"] = get_avg(raw_pose_root_change_buffer)
    avg_metrics_log["Avg Filtered Pose Root Change"] = get_avg(filtered_pose_root_change_buffer)
    avg_metrics_log["Avg Raw Trans Change (mm)"] = get_avg(raw_trans_change_buffer)
    avg_metrics_log["Avg Filtered Trans Change (mm)"] = get_avg(filtered_trans_change_buffer)
    avg_metrics_log["Avg Raw Betas Variance"] = get_avg(raw_betas_variance_buffer)
    avg_metrics_log["Avg Filtered Betas Variance"] = get_avg(filtered_betas_variance_buffer)

    csv_file_already_exists = os.path.isfile(CSV_FILENAME_RELAY)
    try:
        with open(CSV_FILENAME_RELAY,'a',newline='') as csvfile_out:
            csv_dict_writer = csv.DictWriter(csvfile_out, fieldnames=CSV_FIELDNAMES_RELAY)
            if not csv_file_already_exists: csv_dict_writer.writeheader()
            formatted_row = {
                k:(f"{v:.4f}" if isinstance(v,float) and not np.isnan(v) else ("NaN" if isinstance(v,float) else v)) 
                for k,v in avg_metrics_log.items()
            }
            csv_dict_writer.writerow(formatted_row)
        success_log_msg = f"Relay Avg Metrics for '{avg_metrics_log['Condition']}' (Filter: {avg_metrics_log['FilterSetting']}) saved."
        print(success_log_msg)
        if gui_root_relay and hasattr(gui_root_relay,'status_label_var'): gui_root_relay.status_label_var.set(success_log_msg)
    except Exception as e_csv:
        csv_error_msg = f"Relay CSV Write Error: {e_csv}"; print(csv_error_msg); traceback.print_exc()
        if gui_root_relay and hasattr(gui_root_relay,'status_label_var'): gui_root_relay.status_label_var.set(csv_error_msg)

# --- Modified handle_demo_connection with Dynamic Filtering ---
def handle_demo_connection(conn):
    global latest_smpl_data_for_viewer, client_has_connected, stop_all_threads, relay_metric_data_queue
    global active_filter_settings, filter_settings_lock 

    print("Relay: Demo client connection established.")
    client_has_connected.set() 
    conn.settimeout(10.0) 

    local_filters = {"poses_body": None, "poses_root": None, "betas": None, "trans": None}
    local_filter_params_used = {"poses_body": None, "poses_root": None, "betas": None, "trans": None}
    
    prev_raw_data_for_comp_handler = {"valid_for_comparison": False}
    prev_filtered_data_for_comp_handler = {"valid_for_comparison": False}
    last_packet_processing_time = time.time()

    while not stop_all_threads.is_set():
        try:
            received_data_packet = recv_message(conn) 
            timestamp_packet_received = time.time() 
            
            time_since_last_packet_processed = timestamp_packet_received - last_packet_processing_time
            current_relay_packet_fps_handler = 1.0 / time_since_last_packet_processed if time_since_last_packet_processed > 1e-6 else 0.0
            last_packet_processing_time = timestamp_packet_received

            raw_data_for_metrics_dict = None
            filtered_data_for_metrics_dict = None 
            filter_latency_current_frame_ms = 0.0

            if isinstance(received_data_packet, dict) and all(k in received_data_packet for k in ["poses_body", "poses_root", "betas", "trans"]):
                raw_pb_np = received_data_packet['poses_body'].reshape(POSE_BODY_DIM_CONFIG).astype(np.float64)
                raw_pr_np = received_data_packet['poses_root'].reshape(3).astype(np.float64)
                raw_be_np = received_data_packet['betas'].reshape(-1)[:NUM_BETAS_CONFIG].astype(np.float64)
                raw_tr_np = received_data_packet['trans'].reshape(3).astype(np.float64) 
                raw_data_for_metrics_dict = {"poses_body":raw_pb_np, "poses_root":raw_pr_np, "betas":raw_be_np, "trans":raw_tr_np}

                filter_processing_start_time = time.time()
                
                data_streams_to_filter_map = {
                    "poses_body": raw_pb_np, "poses_root": raw_pr_np,
                    "betas": raw_be_np, "trans": raw_tr_np
                }
                current_packet_filtered_data = {} 

                with filter_settings_lock: 
                    global_desired_filter_config = {k: v.copy() if isinstance(v,dict) else v for k,v in active_filter_settings.items()}
                
                current_filter_type = global_desired_filter_config.get("type", "none")

                for stream_key, raw_stream_data_np in data_streams_to_filter_map.items():
                    stream_specific_settings = global_desired_filter_config.get(stream_key, {}) # For OneEuro
                    common_params_for_stream_type = global_desired_filter_config.get("common_params", {}) # For new filters

                    if not stream_specific_settings.get("enabled", False) or current_filter_type == "none":
                        current_packet_filtered_data[stream_key] = raw_stream_data_np.copy()
                        local_filter_params_used[stream_key] = {"type": "none", "enabled": False}
                        local_filters[stream_key] = None 
                        continue

                    filter_params_for_init = {"type": current_filter_type, "enabled": True} # Base params for comparison
                    if current_filter_type == "one_euro":
                        filter_params_for_init.update({
                            "min_cutoff": stream_specific_settings.get("min_cutoff", 1.0),
                            "beta": stream_specific_settings.get("beta", 0.0),
                            "d_cutoff": stream_specific_settings.get("d_cutoff", 1.0)
                        })
                    elif current_filter_type == "single_exp":
                        filter_params_for_init["alpha"] = common_params_for_stream_type.get("alpha", 0.5)
                    elif current_filter_type == "double_exp":
                        filter_params_for_init.update({
                            "alpha": common_params_for_stream_type.get("alpha", 0.5),
                            "beta_trend": common_params_for_stream_type.get("beta_trend", 0.5)
                        })
                    elif current_filter_type == "moving_avg":
                        filter_params_for_init["window_size"] = common_params_for_stream_type.get("window_size", 5)
                    
                    needs_reinit = (
                        local_filters[stream_key] is None or
                        local_filter_params_used[stream_key] != filter_params_for_init 
                    )

                    if needs_reinit:
                        print(f"Relay: Re-initializing filter for '{stream_key}' with type '{current_filter_type}'. Params: {filter_params_for_init}")
                        # Pass only relevant params to constructor
                        constructor_params = {k:v for k,v in filter_params_for_init.items() if k not in ['type', 'enabled']}
                        if current_filter_type == "one_euro":
                            local_filters[stream_key] = OneEuroFilter(raw_stream_data_np, t0=timestamp_packet_received, **constructor_params)
                        elif current_filter_type == "single_exp":
                            local_filters[stream_key] = SingleExponentialFilter(raw_stream_data_np, t0=timestamp_packet_received, **constructor_params)
                        elif current_filter_type == "double_exp":
                            local_filters[stream_key] = DoubleExponentialFilter(raw_stream_data_np, t0=timestamp_packet_received, **constructor_params)
                        elif current_filter_type == "moving_avg":
                            local_filters[stream_key] = MovingAverageFilter(raw_stream_data_np, t0=timestamp_packet_received, **constructor_params)
                        local_filter_params_used[stream_key] = filter_params_for_init.copy()

                    if local_filters[stream_key] is not None:
                         current_packet_filtered_data[stream_key] = local_filters[stream_key](timestamp_packet_received, raw_stream_data_np)
                    else: 
                         current_packet_filtered_data[stream_key] = raw_stream_data_np.copy()

                filter_processing_end_time = time.time()
                filter_latency_current_frame_ms = (filter_processing_end_time - filter_processing_start_time) * 1000
                filtered_data_for_metrics_dict = current_packet_filtered_data

                with data_lock_viewer: 
                    latest_smpl_data_for_viewer = {
                        "poses_body": current_packet_filtered_data["poses_body"].astype(np.float32).reshape(1, POSE_BODY_DIM_CONFIG),
                        "poses_root": current_packet_filtered_data["poses_root"].astype(np.float32).reshape(1, 3),
                        "betas": current_packet_filtered_data["betas"].astype(np.float32).reshape(1, NUM_BETAS_CONFIG),
                        "trans": current_packet_filtered_data["trans"].astype(np.float32).reshape(1, 3),
                    }
            else: 
                print(f"Relay: Invalid data format from demo: {type(received_data_packet)}")
                prev_raw_data_for_comp_handler["valid_for_comparison"] = False
                prev_filtered_data_for_comp_handler["valid_for_comparison"] = False

            if raw_data_for_metrics_dict: 
                metric_data_packet_to_queue = {
                    "timestamp": timestamp_packet_received, "raw_data": raw_data_for_metrics_dict,
                    "filtered_data": filtered_data_for_metrics_dict if filtered_data_for_metrics_dict else raw_data_for_metrics_dict.copy(),
                    "prev_raw_data_for_comp": prev_raw_data_for_comp_handler.copy(), 
                    "prev_filtered_data_for_comp": prev_filtered_data_for_comp_handler.copy(),
                    "filter_latency_ms": filter_latency_current_frame_ms,
                    "relay_packet_processing_fps": current_relay_packet_fps_handler
                }
                try: relay_metric_data_queue.put_nowait(metric_data_packet_to_queue)
                except queue.Full: print("Warning: Relay metric queue full. Dropping packet.")

                prev_raw_data_for_comp_handler = raw_data_for_metrics_dict.copy(); prev_raw_data_for_comp_handler["valid_for_comparison"] = True
                if filtered_data_for_metrics_dict:
                    prev_filtered_data_for_comp_handler = filtered_data_for_metrics_dict.copy(); prev_filtered_data_for_comp_handler["valid_for_comparison"] = True
                else: 
                    prev_filtered_data_for_comp_handler = raw_data_for_metrics_dict.copy(); prev_filtered_data_for_comp_handler["valid_for_comparison"] = True
            else: 
                 prev_raw_data_for_comp_handler["valid_for_comparison"] = False
                 prev_filtered_data_for_comp_handler["valid_for_comparison"] = False

        except socket.timeout: continue 
        except (ConnectionResetError, BrokenPipeError, EOFError, OSError, ValueError, struct.error, pickle.UnpicklingError) as e_sock_data:
            print(f"Relay: Socket/Data error from demo: {e_sock_data}. Disconnecting."); break 
        except Exception as e_handler: 
            print(f"Relay: Unexpected error in demo handler: {e_handler}"); traceback.print_exc(); break
    
    print("Relay: Demo client connection handler finished.")
    client_has_connected.clear() 
    try: conn.shutdown(socket.SHUT_RDWR) 
    except OSError: pass 
    finally: conn.close()

# --- Socket Server Thread (listens for demo.py connections) ---
def socket_server_thread_relay():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    try:
        server_socket.bind((RELAY_LISTEN_HOST, RELAY_LISTEN_PORT))
        server_socket.listen(1) 
        print(f"Relay server listening on {RELAY_LISTEN_HOST}:{RELAY_LISTEN_PORT} for demo.py connections.")
        while not stop_all_threads.is_set():
            server_socket.settimeout(1.0) 
            try:
                conn, addr = server_socket.accept() 
                print(f"Relay: Accepted connection from demo client: {addr}")
                client_handler_thread = threading.Thread(target=handle_demo_connection, args=(conn,), daemon=True)
                client_handler_thread.start()
            except socket.timeout: continue 
            except Exception as e_accept: 
                print(f"Relay: Error accepting new connection: {e_accept}"); time.sleep(1) 
    except Exception as e_server_start: 
        print(f"Relay: FATAL ERROR - Failed to start relay server: {e_server_start}"); traceback.print_exc()
        stop_all_threads.set() 
    finally: 
        print("Relay: Closing main relay server socket."); server_socket.close()

# --- Main Script Execution ---
if __name__ == "__main__":
    relay_gui_thread = threading.Thread(target=setup_gui_relay, daemon=True)
    relay_gui_thread.start()
    relay_metric_thread_instance = threading.Thread(target=metrics_calculation_worker_relay, daemon=True)
    relay_metric_thread_instance.start()
    relay_server_thread = threading.Thread(target=socket_server_thread_relay, daemon=True)
    relay_server_thread.start()

    print(f"Relay: Attempting to connect to AITViewer server at {VIEWER_HOST}:{VIEWER_PORT}...")
    v = RemoteViewer(host=VIEWER_HOST, port=VIEWER_PORT, verbose=False) 
    if not v.connected:
        print(f"Relay: FATAL ERROR - Could not connect to AITViewer server."); stop_all_threads.set() 
        if relay_server_thread.is_alive(): relay_server_thread.join(timeout=1.0)
        if relay_metric_thread_instance and relay_metric_thread_instance.is_alive(): relay_metric_thread_instance.join(timeout=1.0)
        if relay_gui_thread and relay_gui_thread.is_alive(): relay_gui_thread.join(timeout=1.0)
        sys.exit(1) 
    print("Relay: Successfully connected to AITViewer server.")

    smpl_render_sequence_aitviewer = None 
    try:
        print("Relay: Creating initial SMPL sequence for AITViewer...")
        init_poses_body_np = np.zeros((1, POSE_BODY_DIM_CONFIG),dtype=np.float32)
        init_poses_root_np = np.zeros((1,3),dtype=np.float32)
        init_betas_np = np.zeros((1,NUM_BETAS_CONFIG),dtype=np.float32)
        init_trans_np = np.array([[0.,0.,0.]],dtype=np.float32) 
        smpl_render_sequence_aitviewer = RemoteSMPLSequence(
            viewer=v, poses_body=init_poses_body_np, poses_root=init_poses_root_np, 
            betas=init_betas_np, trans=init_trans_np,
            gender='neutral', model_type='smpl', 
            name="Relayed Filtered SMPL", color=(0.2,0.6,1.0,1.0) 
        )
        print(f"Relay: Initial SMPL sequence for AITViewer created with remote_uid: {smpl_render_sequence_aitviewer.uid}")
    except Exception as e_ait_smpl:
        print(f"Relay: FATAL ERROR creating RemoteSMPLSequence: {e_ait_smpl}"); traceback.print_exc()
        if v.connected: v.close_connection();
        stop_all_threads.set()
        if relay_server_thread.is_alive(): relay_server_thread.join(timeout=1.0)
        if relay_metric_thread_instance and relay_metric_thread_instance.is_alive(): relay_metric_thread_instance.join(timeout=1.0)
        if relay_gui_thread and relay_gui_thread.is_alive(): relay_gui_thread.join(timeout=1.0)
        sys.exit(1)

    print("Relay: Starting main relay loop (sending data to AITViewer)...")
    last_viewer_update_sent_time = time.time()
    viewer_target_update_interval = 1.0 / 30.0 

    try:
        while not stop_all_threads.is_set():
            data_to_send_to_viewer = None 
            with data_lock_viewer: 
                if latest_smpl_data_for_viewer:
                    data_to_send_to_viewer = latest_smpl_data_for_viewer.copy() 
                    latest_smpl_data_for_viewer = None 

            if data_to_send_to_viewer and smpl_render_sequence_aitviewer: 
                try:
                    smpl_render_sequence_aitviewer.update_frames(
                        poses_body=data_to_send_to_viewer['poses_body'], frames=0, 
                        poses_root=data_to_send_to_viewer['poses_root'],
                        betas=data_to_send_to_viewer['betas'],
                        trans=data_to_send_to_viewer['trans']
                    )
                    last_viewer_update_sent_time = time.time() 
                except KeyError as e_key_viewer: 
                    print(f"Relay: ERROR - Data for AITViewer missing key: {e_key_viewer}")
                except Exception as e_update_viewer: 
                    print(f"Relay: ERROR updating AITViewer sequence: {e_update_viewer}"); traceback.print_exc()
            
            time_since_last_viewer_send = time.time() - last_viewer_update_sent_time
            sleep_time_for_viewer_loop = max(0, viewer_target_update_interval - time_since_last_viewer_send) \
                                         if data_to_send_to_viewer else \
                                         (0.1 if not client_has_connected.is_set() else 0.02) 
            time.sleep(sleep_time_for_viewer_loop)

            if not relay_server_thread.is_alive() and not stop_all_threads.is_set():
                print("Relay: CRITICAL ERROR - Socket server thread died."); stop_all_threads.set(); break 
            if not v.connected and not stop_all_threads.is_set(): 
                print("Relay: CRITICAL ERROR - AITViewer connection lost."); stop_all_threads.set(); break

    except KeyboardInterrupt: print("\nRelay: Keyboard interrupt.")
    finally: 
        print("Relay: Initiating shutdown..."); stop_all_threads.set() 
        if relay_metric_thread_instance and relay_metric_thread_instance.is_alive():
            try: relay_metric_data_queue.put_nowait(None) 
            except queue.Full: pass 
        if v.connected: v.close_connection() 
        print("Relay: Waiting for threads...")
        if relay_server_thread.is_alive(): relay_server_thread.join(timeout=2.0)
        if relay_metric_thread_instance and relay_metric_thread_instance.is_alive(): relay_metric_thread_instance.join(timeout=2.0)
        if relay_gui_thread and relay_gui_thread.is_alive():
            if gui_root_relay: 
                try: 
                    gui_root_relay.destroy()
                except: pass
            relay_gui_thread.join(timeout=2.0)
        if relay_server_thread.is_alive(): print("Warning: Relay server thread did not exit.")
        if relay_metric_thread_instance and relay_metric_thread_instance.is_alive(): print("Warning: Relay metric thread did not exit.")
        if relay_gui_thread and relay_gui_thread.is_alive(): print("Warning: Relay GUI thread did not exit.")
        print("Relay: Script finished.")
