# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
# custom_server.py - Cleaned up for SMPL relay
import numpy as np
import traceback # Already included

# --- AITViewer Imports ---
try:
    from aitviewer.configuration import CONFIG as C
    from aitviewer.remote.message import Message
    from aitviewer.scene.node import Node
    from aitviewer.viewer import Viewer
except ImportError as e:
    print(f"ERROR: Failed to import aitviewer components: {e}")
    print("Ensure aitviewer is installed correctly in your environment.")
    import sys
    sys.exit(1)

# No custom message types needed anymore for the relay
# MSG_SET_POSITION = Message.USER_MESSAGE.value + 1 # REMOVED

class ControllableViewer(Viewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized ControllableViewer (SMPL Relay Mode)")

    def process_message(self, type: Message, remote_uid: int, args: tuple, kwargs: dict, client):
        """
        Overrides the default message processing.
        Only passes messages to the default handler now.
        """
        # --- REMOVED Custom MSG_SET_POSITION Handling ---
        # The 'if type == MSG_SET_POSITION:' block is gone.

        # Pass ALL messages (SMPL creation, UPDATE_FRAMES, etc.) to the default handler
        try:
             if self.server:
                # Pass the original args tuple and kwargs dict
                self.server.process_message(type, remote_uid, args, kwargs, client)
             else:
                 print(f"Cannot process standard message type {type}: Server not available")
        except Exception as e:
            # Print the custom prefix AND the full traceback
            print(f"Exception while processing standard message: type = {type}, remote_uid = {remote_uid}:")
            traceback.print_exc()


# --- Main part of the server script ---
if __name__ == "__main__":
    print("Starting Custom Viewer Server...")
    try:
        C.update_conf({"server_enabled": True, "server_port": 8417})
        v = ControllableViewer(title="SMPL Viewer (via Relay)")
        v.scene.floor.enabled = True 
        print(f"Viewer Server running on port {C.server_port}. Waiting for relay client (remote_relay.py)...")
        v.run()
    except Exception as e:
         print(f"ERROR starting viewer server: {e}")
         traceback.print_exc()
    finally:
        print("Viewer server stopped.")