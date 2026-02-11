import asyncio
import websockets

async def connect():
    uri = "ws://10.160.217.171:8000"
    async with websockets.connect(uri) as websocket:
        print("Connected successfully")
        await websocket.send("Hello")
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(connect())