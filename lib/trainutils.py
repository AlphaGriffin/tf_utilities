import os


def gamepadImageMatcher (def path):
	'''SAW -- matches gamepad csv data rows to images based on timestamps
		Returns two matching arrays'''
	
	# Open CSV for reading
	csv_path = os.path.join(path, "data.csv")
	print ("about to open: {}".format(csv_path))
	csv_io = open(csv_path, 'r')
	print ("csv successfully opened")
	
	# Convert to a true array
	csv = []
	for line in csv_io:
		# Split the string into array and trim off any whitespace/newlines in elements
		csv.append([item.strip() for item in line.split(',')])

	if not csv:
		print ("CSV HAS NO DATA");
		return None, None
	
	# Get list of images in directory and sort it
	all_files = os.listdir(path)
	print ("all_files: {}".format(all_files))
	#images = [image for image in all_files if any(image.endswith('.png'))]
	images = []
	for filename in all_files:
		if filename.endswith('.png'):
			images.append(filename)
	images = sorted(images)
	print ("sorted images list: {}".format(images))

	if not images:
		print ("FOUND NO IMAGES");
		return None, None

	# We're going to build up 2 arrays of matching size:
	keep_csv = []
	keep_images = []
	
	# Prime the pump (queue)...
	prev_line = csv.pop(0)
	prev_csvtime = int(prev_line[0])
	print ("line 0: {}".format(prev_line))
	
#            prev_image = images[0]
#            prev_imgtime = 0
	
	# Find next matching image
	while images:
		imgfile = images[0]
		print ("imgfile: {}".format(imgfile))
		
		# Get image time:
		#     Cut off the "gamename-" from the front and the ".png" from the end
		hyphen = imgfile.rfind('-') # Get last index of '-'
		if hyphen < 0:
			print ("UNEXPECTED FILENAME, ABORTING: {}".format(imgfile))
			break
		imgtime = int(imgfile[hyphen+1:-4]) # cut it out!
		print ("imgtime: {}".format(imgtime))
		
		lastKeptWasImage = False # Did we last keep an image, or a line?
	
		if imgtime > prev_csvtime:
			print ("keeping image: {}".format(imgtime))
			keep_images.append(imgfile)
			del images[0]
			lastKeptWasImage = True
			
			# We just kept an image, so we need to keep a
			# corresponding input row too
			while csv:
				line = csv.pop(0)
				print ("line: {}".format(line))
				
				# Get CSV line timestamp:
				csvtime = int(line[0])
				print ("csvtime: {}".format(str(csvtime)))
				
				if csvtime >= imgtime:
					# We overshot the input queue... ready to
					# keep the previous data line
					print ("keeping line: {}".format(prev_csvtime))
					keep_csv.append(prev_line)
					lastKeptWasImage = False
					
				prev_line = line
				prev_csvtime = csvtime
				
				if csvtime >= imgtime:
					break;
					
			if not csv:
				print ("OUT OF CSV DATA")
				if lastKeptWasImage:
					print ("keeping line: {}".format(prev_csvtime))
					keep_csv.append(prev_line)
				break
				
		else:
			print ("dropping image: {}".format(imgtime))
			del images[0]
		
	print ("len(keep_data): {}".format(len(keep_csv)))
	print ("len(keep_images): {}".format(len(keep_images)))
#	self.print_console("MATCHED RESULTS COUNT: {} <--> {}".format(len(keep_csv),len(keep_images)))

	return keep_csv, keep_images
