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

# --- AITViewer Imports ---
try:
    from aitviewer.remote.renderables.smpl import RemoteSMPLSequence
    from aitviewer.remote.viewer import RemoteViewer
    from aitviewer.remote.message import Message
except ImportError as e:
     print(f"ERROR: Failed to import aitviewer components: {e}")
     print("Ensure aitviewer is installed correctly in your environment.")
     sys.exit(1)

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
    """Handles receiving data from the demo.py client."""
    global latest_smpl_data
    print("Relay: Demo client connected.")
    client_connected.set() # Signal that a client has connected
    conn.settimeout(10.0) # Increased timeout for potentially slower demo processing

    data_buffer = b'' # Buffer for accumulating data chunks

    while not stop_threads.is_set():
        try:
            # # --- Receiving logic with size prefix ---
            # msg_len = None
            # # First, try to read the 4-byte length prefix if not already in buffer
            # while len(data_buffer) < 4:
            #     chunk = conn.recv(4 - len(data_buffer))
            #     if not chunk:
            #         print("Relay: Demo client disconnected (receiving length prefix).")
            #         stop_threads.set() # Signal main thread to exit
            #         return # Exit handler thread
            #     data_buffer += chunk
            #     # Check stop_threads periodically while waiting for prefix
            #     if stop_threads.is_set(): return

            # # We have the length prefix, unpack it
            # msg_len = struct.unpack('>I', data_buffer[:4])[0]
            # data_buffer = data_buffer[4:] # Remove length prefix from buffer

            # # Read the message data until full message is received
            # while len(data_buffer) < msg_len:
            #     chunk = conn.recv(min(4096, msg_len - len(data_buffer))) # Read in chunks
            #     if not chunk:
            #         print("Relay: Demo client disconnected (receiving data).")
            #         stop_threads.set() # Signal main thread to exit
            #         return # Exit handler thread
            #     data_buffer += chunk
            #      # Check stop_threads periodically while waiting for data
            #     if stop_threads.is_set(): return

            # # We have the full message, process it
            # serialized_data = data_buffer[:msg_len]
            # data_buffer = data_buffer[msg_len:] # Keep remaining buffer for next message

            # # Deserialize
            # received_data = pickle.loads(serialized_data)
            
            received_data = recv_message(conn)

            # Validate data (basic check)
            if isinstance(received_data, dict) and "poses_body" in received_data:
                 with data_lock:
                    latest_smpl_data = received_data
                 # print("Relay: Received valid SMPL data update.") # Verbose
            else:
                 print(f"Relay: Received invalid data format: {type(received_data)}")

        except socket.timeout:
            # print("Relay: Socket recv timeout, check if demo.py is still running/sending.")
            # If demo stops sending, we'll keep timing out here until disconnected
            continue
        except (ConnectionResetError, BrokenPipeError, EOFError, OSError) as e:
            print(f"Relay: Socket error receiving data: {e}. Demo likely disconnected.")
            break # Exit loop on connection error
        except pickle.UnpicklingError as e:
            print(f"Relay: Pickle error receiving data: {e}")
            break # Stop on bad data
        except struct.error as e:
             print(f"Relay: Struct unpack error (likely bad length prefix): {e}")
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
        initial_pose_body = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
        initial_pose_root = np.zeros((1, 3), dtype=np.float32)
        initial_betas = np.zeros((1, NUM_BETAS), dtype=np.float32)
        initial_trans = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        smpl_sequence = RemoteSMPLSequence(
            v,
            poses_body=initial_pose_body,
            poses_root=initial_pose_root,
            betas=initial_betas,
            trans=initial_trans,
            gender='neutral',
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
    last_update_time = 0
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
                    # Extract data (ensure shapes are correct, add batch dim if needed)
                    pb = current_data['poses_body'].reshape(1, POSE_BODY_DIM)
                    pr = current_data['poses_root'].reshape(1, 3)
                    be = current_data['betas'].reshape(1, -1)[:, :NUM_BETAS]
                    tr = current_data['trans'].reshape(1, 3)

                    # Send update to viewer server
                    smpl_sequence.update_frames(
                        poses_body=pb,
                        frames=0,
                        poses_root=pr,
                        betas=be,
                        trans=tr
                    )
                    # print("Relay: Sent update_frames to viewer.") # Verbose
                    last_update_time = time.time()

                except KeyError as e:
                     print(f"Relay: ERROR - Received data missing key: {e}")
                except Exception as e:
                    print(f"Relay: ERROR processing/sending received data: {e}")
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