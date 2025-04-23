# preprocessing/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import PreprocessingStep

class StatusConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.channel_layer.group_add("status_updates", self.channel_name)
        
        # Send initial state
        steps = await self.get_all_steps()
        await self.send(text_data=json.dumps({
            'type': 'initial_status',
            'steps': steps
        }))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("status_updates", self.channel_name)

    async def status_update(self, event):
        await self.send(text_data=json.dumps(event))

    @database_sync_to_async
    def get_all_steps(self):
        return list(PreprocessingStep.objects.all().values())