from ultralytics import YOLO

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
import argparse

import time
start_time = time.time()


from pythonosc import udp_client

# Store the track history
trackid_time = {}

stored_tracks = {}



parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1",
	help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=5006,
	help="The port the OSC server is listening on")
args = parser.parse_args()

client = udp_client.SimpleUDPClient(args.ip, args.port)

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official detection model


# Track with the model
results = model.track(source="0", classes=0, show=True ,stream=True, persist=True)

		
# Process results list
for result in results:
	boxes = result.boxes  # Boxes object for bbox outputs
	
	for box in boxes:
		track_id = box.id
		#getting new detections and storing the time they occur
		if track_id != None:
			track_id = box.id.int().cpu().tolist()[0]
			print(track_id)
			if str(track_id) not in trackid_time:
				tracktime = time.time() - start_time
				newItem = {str(track_id): tracktime}
				trackid_time.update(newItem)
			
	# store tracking data into dict
	for box in boxes:
		
		if box.id != None:
			id = box.id.int().cpu().tolist()[0]
			tracktime = trackid_time.get(str(id))
			
			#only output results that have stuck around for a second
			if tracktime < ((time.time() - start_time) - 1):
				vals = []
				vals.append(id)
				vals.append(box.xywhn.cpu().tolist())
				vals.append((time.time() - start_time))
				vals.append(tracktime)
				vals.append(int(box.cls))

				trackdata = {str(id) : vals}
				stored_tracks.update(trackdata)
				
	index = 0
	removeids = []

	#go through stored data and get ids to remove. inactive for 1 second
	for id in stored_tracks:
			trackdata= stored_tracks.get(str(id))
			if((time.time() - start_time) - 1 > trackdata[2]):
				removeids.append(str(id))

	#remove dict elements
	for id in removeids:
		stored_tracks.pop(id)
		trackid_time.pop(id)
		
	#print to table
	for id in stored_tracks:
		trackdata= stored_tracks.get(str(id))
		client.send_message("/trackingdata",trackdata)
		print(trackdata)
		index += 1
		




