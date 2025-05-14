# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
# Modified remote_relay.py acting as RELAY (Server for demo.py, Client for viewer)
import sys
import time
import numpy as np
import socket
import pickle
import zlib
import struct
import threading
import traceback # For detailed error printing
import math # Needed for math.pi in One Euro Filter

# --- AITViewer Imports ---
try:
    from aitviewer.remote.renderables.smpl import RemoteSMPLSequence
    from aitviewer.remote.viewer import RemoteViewer
    from aitviewer.remote.message import Message
except ImportError as e:
     print(f"ERROR: Failed to import aitviewer components: {e}")
     print("Ensure aitviewer is installed correctly in your environment.")
     sys.exit(1)

# --- One Euro Filter Implementation (NumPy compatible) ---
def smoothing_factor(t_e, cutoff):
    """Compute the smoothing factor (scalar inputs)."""
    # t_e: elapsed time (scalar)
    # cutoff: cutoff frequency (scalar)
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    """Apply exponential smoothing (scalar 'a', numpy array inputs)."""
    # a: smoothing factor (scalar)
    # x: current raw value (NumPy array)
    # x_prev: previous filtered value (NumPy array)
    # Note: NumPy handles scalar 'a' broadcasting correctly
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    def __init__(self, x0, t0=None, dx0=None, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter for NumPy arrays."""
        # Ensure initial value is a numpy array
        if not isinstance(x0, np.ndarray):
             raise ValueError("Initial value x0 must be a numpy array")

        # The parameters (scalars).
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        # Store the shape of the data
        self.data_shape = x0.shape

        # Previous values (will be NumPy arrays). Store copies.
        self.x_prev = x0.astype(np.float64) # Use float64 for potentially better precision

        # Initialize previous speed
        if dx0 is None:
             self.dx_prev = np.zeros_like(x0, dtype=np.float64)
        elif isinstance(dx0, np.ndarray) and dx0.shape == x0.shape:
             self.dx_prev = dx0.astype(np.float64)
        else:
             raise ValueError("dx0 must be None or a numpy array of the same shape as x0")

        # Previous timestamp. Use current time if t0 is None.
        self.t_prev = float(t0) if t0 is not None else time.time()


    def __call__(self, t, x):
        """Compute the filtered signal."""
        # t: current timestamp (scalar)
        # x: current raw value (NumPy array)

        # Ensure input is a numpy array and shape matches
        if not isinstance(x, np.ndarray):
             raise ValueError("Input value 'x' must be a numpy array")
        if x.shape != self.data_shape:
            # Handle shape mismatch: potentially reset filter state or raise error
            # Resetting is safer for changing numbers of people/detections
            print(f"Warning: OneEuroFilter input shape mismatch. Expected {self.data_shape}, got {x.shape}. Resetting filter.")
            self.data_shape = x.shape # Update shape
            self.x_prev = x.astype(np.float64) # Reset state with new data
            self.dx_prev = np.zeros_like(x, dtype=np.float64)
            self.t_prev = float(t)
            return x.copy() # Return raw value for this frame

        # Ensure input data type matches internal filter type
        x_array = x.astype(np.float64)
        # Ensure timestamp is float
        t_scalar = float(t)

        t_e = t_scalar - self.t_prev

        # Handle edge case: elapsed time is zero or negative
        if t_e <= 0:
             # Return previous filtered value to maintain smoothness
             return self.x_prev.copy()

        # The filtered derivative of the signal (NumPy array).
        # Note: smoothing_factor expects scalar inputs
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x_array - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal (NumPy array).
        # Calculate speed magnitude (scalar) using L2 norm
        speed_magnitude = np.linalg.norm(dx_hat)

        # Calculate dynamic cutoff frequency (scalar)
        cutoff = self.min_cutoff + self.beta * speed_magnitude

        # Calculate smoothing factor for the signal (scalar)
        # Note: smoothing_factor expects scalar inputs
        a = smoothing_factor(t_e, cutoff)

        # Filter the signal (NumPy array)
        x_hat = exponential_smoothing(a, x_array, self.x_prev)

        # Memorize the previous values (NumPy arrays). Store copies.
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t_scalar

        return x_hat.copy() # Return a copy of the smoothed array

# --- End One Euro Filter Implementation ---


# --- Configuration ---
VIEWER_HOST = "localhost" # Host where custom_smpl_server.py is running
VIEWER_PORT = 8417       # Port where custom_smpl_server.py is running

RELAY_HOST = 'localhost' # Host for this script to listen on
RELAY_PORT = 9999       # Port for this script to listen on (must match demo.py)

NUM_BETAS = 10
POSE_BODY_DIM = 69 # For basic 'smpl' model

# --- Global variable to store latest received data ---
latest_smpl_data = None
data_lock = threading.Lock() # To safely access latest_smpl_data
client_connected = threading.Event() # To signal when demo.py connects
stop_threads = threading.Event() # To signal threads to stop

target_fps = 30
frame_interval = 1.0
last_send = time.time()

def recv_n_bytes(sock, n):
    """
    Read exactly n bytes from sock.
    Raises ConnectionError if the socket closes before n bytes are read.
    """
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            # peer closed connection or error
            raise ConnectionError(f"Socket closed with {n - len(buf)} bytes left to read")
        buf += chunk
    return buf

MAGIC = b'SMPL'    # must match sender

def recv_message(conn):
    # 1) Read header
    hdr = recv_n_bytes(conn, 4 + 4 + 4)   # magic + length + crc32
    magic, length, sent_crc = struct.unpack('>4s I I', hdr)
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r}")

    # 2) Read the payload
    payload = recv_n_bytes(conn, length)

    # 3) Check CRC
    if zlib.crc32(payload) & 0xFFFFFFFF != sent_crc:
        raise ValueError("Checksum mismatch — data corrupted in transit")

    # 4) Unpickle
    return pickle.loads(payload)

def handle_demo_connection(conn):
    """Handles receiving data from the demo.py client and applies smoothing."""
    global latest_smpl_data
    print("Relay: Demo client connected.")
    client_connected.set() # Signal that a client has connected
    conn.settimeout(10.0) # Increased timeout for potentially slower demo processing

    # Initialize One Euro Filters for this connection
    # These will be initialized with the first frame's data
    poses_root_filter = None
    poses_body_filter = None
    betas_filter = None
    trans_filter = None

    while not stop_threads.is_set():
        try:
            received_data = recv_message(conn)
            current_time = time.time() # Get timestamp for filtering

            if isinstance(received_data, dict) and "poses_body" in received_data and \
               "poses_root" in received_data and "betas" in received_data and "trans" in received_data:

                 # Extract raw data as 1D arrays (remove batch dimension)
                 # Ensure data types are float for filtering
                 raw_poses_body = received_data['poses_body'].reshape(POSE_BODY_DIM).astype(np.float64)
                 raw_poses_root = received_data['poses_root'].reshape(3).astype(np.float64)
                 raw_betas = received_data['betas'].reshape(-1)[:NUM_BETAS].astype(np.float64)
                 raw_trans = received_data['trans'].reshape(3).astype(np.float64)

                 # --- Apply One Euro Filter ---
                 smoothed_poses_body = raw_poses_body # Default to raw if filter not initialized
                 smoothed_poses_root = raw_poses_root
                 smoothed_betas = raw_betas
                 smoothed_trans = raw_trans

                 if poses_body_filter is None:
                     # Initialize filters with the first frame's data and current time
                     # Tune min_cutoff, beta, and d_cutoff based on desired smoothness/responsiveness
                     # These are example values; adjust as needed.
                     poses_body_filter = OneEuroFilter(raw_poses_body, t0=current_time, min_cutoff=1.0, beta=0.5)
                     poses_root_filter = OneEuroFilter(raw_poses_root, t0=current_time, min_cutoff=1.0, beta=0.5)
                     betas_filter = OneEuroFilter(raw_betas, t0=current_time, min_cutoff=0.5, beta=0.2)
                     trans_filter = OneEuroFilter(raw_trans, t0=current_time, min_cutoff=1.0, beta=0.7)
                     print("Relay: Initialized One Euro Filters with first frame data.")
                 else:
                     # Process subsequent frames with the filters
                     smoothed_poses_body = poses_body_filter(current_time, raw_poses_body)
                     smoothed_poses_root = poses_root_filter(current_time, raw_poses_root)
                     smoothed_betas = betas_filter(current_time, raw_betas)
                     smoothed_trans = trans_filter(current_time, raw_trans)

                 # Package smoothed data back into the dictionary format (add batch dimension back)
                 # Ensure data types are float32 for sending (matching original format)
                 with data_lock:
                     latest_smpl_data = {
                         "poses_body": smoothed_poses_body.astype(np.float32).reshape(1, POSE_BODY_DIM),
                         "poses_root": smoothed_poses_root.astype(np.float32).reshape(1, 3),
                         "betas": smoothed_betas.astype(np.float32).reshape(1, NUM_BETAS),
                         "trans": smoothed_trans.astype(np.float32).reshape(1, 3),
                     }
                 # print("Relay: Received, smoothed, and stored data update.") # Verbose

            else:
                 print(f"Relay: Received invalid data format or missing keys: {type(received_data)}")

        except socket.timeout:
            # print("Relay: Socket recv timeout, check if demo.py is still running/sending.")
            continue # Keep trying to receive
        except (ConnectionResetError, BrokenPipeError, EOFError, OSError) as e:
            print(f"Relay: Socket error receiving data: {e}. Demo likely disconnected.")
            break # Exit loop on connection error
        except pickle.UnpicklingError as e:
            print(f"Relay: Pickle error receiving data: {e}")
            break # Stop on bad data
        except struct.error as e:
             print(f"Relay: Struct unpack error (likely bad header): {e}")
             break # Stop on bad data format
        except Exception as e:
            print(f"Relay: Unexpected error handling demo connection: {e}")
            traceback.print_exc()
            break # Exit loop on unexpected error

    print("Relay: Demo client connection handler finished.")
    client_connected.clear()
    try:
        conn.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    finally:
        conn.close()
    # Optionally signal main thread to exit if desired when demo disconnects
    # stop_threads.set()


def socket_server_thread():
    """Runs the socket server to listen for demo.py."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((RELAY_HOST, RELAY_PORT))
        server_socket.listen(1)
        print(f"Relay server listening on {RELAY_HOST}:{RELAY_PORT} for demo.py")

        while not stop_threads.is_set():
            server_socket.settimeout(1.0)
            try:
                conn, addr = server_socket.accept()
                print(f"Relay: Connection accepted from {addr}")
                # Start handler thread
                handler_thread = threading.Thread(target=handle_demo_connection, args=(conn,), daemon=True)
                handler_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                 print(f"Relay: Error accepting connection: {e}")
                 time.sleep(1)

    except Exception as e:
        print(f"Relay: ERROR - Failed to start relay server: {e}")
        traceback.print_exc()
        stop_threads.set() # Signal main thread to exit if server fails
    finally:
        print("Relay: Closing relay server socket.")
        server_socket.close()


# --- Main Script Logic ---
if __name__ == "__main__":

    # --- Start Socket Server Thread ---
    server_thread = threading.Thread(target=socket_server_thread, daemon=True)
    server_thread.start()

    # --- Connect to AITViewer Server ---
    print(f"Relay: Connecting to viewer server at {VIEWER_HOST}:{VIEWER_PORT}...")
    v = RemoteViewer(host=VIEWER_HOST, port=VIEWER_PORT, verbose=False)

    if not v.connected:
        print(f"Relay: ERROR - Could not connect to viewer server. Is custom_smpl_server.py running?")
        stop_threads.set()
        server_thread.join()
        sys.exit(1)
    print("Relay: Connected to viewer server.")

    # --- Create Initial RemoteSMPLSequence ---
    smpl_sequence = None
    try:
        print("Relay: Creating initial SMPL sequence (1 frame T-pose)...")
        # Initial data should match the expected format and dimensions
        initial_pose_body = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
        initial_pose_root = np.zeros((1, 3), dtype=np.float32)
        initial_betas = np.zeros((1, NUM_BETAS), dtype=np.float32)
        initial_trans = np.array([[0.0, 0.0, 0.0]], dtype=np.float32) # Use a neutral initial translation

        smpl_sequence = RemoteSMPLSequence(
            v,
            poses_body=initial_pose_body,
            poses_root=initial_pose_root,
            betas=initial_betas,
            trans=initial_trans,
            gender='neutral', # Assuming neutral model
            model_type='smpl', # Match model expected from demo.py data
            name="Relayed SMPL",
            color=(0.8, 1.0, 0.6, 1.0),
        )
        print(f"Relay: Initial SMPL sequence created with remote_uid: {smpl_sequence.uid}")
    except Exception as e:
        print(f"Relay: ERROR creating RemoteSMPLSequence: {e}")
        traceback.print_exc()
        if v.connected: v.close_connection()
        stop_threads.set()
        server_thread.join()
        sys.exit(1)

    # --- Main Loop: Check for new data and update viewer ---
    print("Relay: Starting relay loop. Waiting for demo.py to connect...")
    last_update_time = time.time()
    update_interval = 1.0 / 30.0 # Target ~30fps updates to viewer

    try:
        while not stop_threads.is_set():
            current_data = None
            with data_lock:
                if latest_smpl_data:
                    current_data = latest_smpl_data.copy() # Copy data to avoid holding lock long
                    latest_smpl_data = None # Consume the data

            if current_data:
                try:
                    # Extract data (already smoothed and in correct shape from handler)
                    pb = current_data['poses_body']
                    pr = current_data['poses_root']
                    be = current_data['betas']
                    tr = current_data['trans']

                    # Send update to viewer server
                    smpl_sequence.update_frames(
                        poses_body=pb,
                        frames=0, # Update frame 0 of the sequence
                        poses_root=pr,
                        betas=be,
                        trans=tr
                    )
                    # print("Relay: Sent update_frames to viewer.") # Verbose
                    last_update_time = time.time()

                except KeyError as e:
                     print(f"Relay: ERROR - Data from handler missing key: {e}")
                except Exception as e:
                    print(f"Relay: ERROR processing/sending received data to viewer: {e}")
                    traceback.print_exc()

            # Sleep briefly
            # Only apply target rate if we actually sent an update
            if current_data:
                time_since_last_update = time.time() - last_update_time
                sleep_time = max(0, update_interval - time_since_last_update)
            else:
                # Wait longer if no data or demo not connected yet
                 sleep_time = 0.1 if not client_connected.is_set() else 0.02
            time.sleep(sleep_time)

            # Check if server thread is still alive (optional robustness)
            if not server_thread.is_alive():
                 print("Relay: ERROR - Socket server thread died unexpectedly.")
                 break


    except KeyboardInterrupt:
        print("\nRelay: Keyboard interrupt received.")
    finally:
        print("Relay: Shutting down...")
        stop_threads.set()
        if v.connected:
            v.close_connection()
        print("Relay: Waiting for server thread to finish...")
        server_thread.join(timeout=2.0)
        print("Relay: script finished.")
