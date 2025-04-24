# api/utils.py
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

def broadcast_update(step_data):
    async_to_sync(get_channel_layer().group_send)(
        "status_updates",
        {"type": "status_update", "step": step_data}
    )