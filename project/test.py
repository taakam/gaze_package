import zmq
ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PUSH)
sock.connect("tcp://127.0.0.1:5557")
sock.send_json({"frame_id": "base_link", "x": 0.74, "y": 0.0, "z": 0.15})