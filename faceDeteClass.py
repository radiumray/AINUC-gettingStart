from openvino.inference_engine import IENetwork, IEPlugin, IECore
import cv2
import numpy as np



class FaceDeteClass:
	def __init__(self, FACEDETECT_XML, FACEDETECT_BIN):

		self.FACEDETECT_XML=FACEDETECT_XML
		self.FACEDETECT_BIN=FACEDETECT_BIN
		self.FACEDETECT_INPUTKEYS = 'data'
		self.FACEDETECT_OUTPUTKEYS = 'detection_out'

		self.ie = IECore()
		#  Read in Graph file (IR) to create network
		self.net = IENetwork(self.FACEDETECT_XML, self.FACEDETECT_BIN)
		# Load the Network using Plugin Device
		self.exec_net = self.ie.load_network(network=self.net, device_name='CPU')
		# exec_net = plugin.load(network=net)
		self.n_facedetect, self.c_facedetect, self.h_facedetect, self.w_facedetect = self.net.inputs[
			self.FACEDETECT_INPUTKEYS].shape


	def image_preprocessing(self, image, n, c, h, w):
		"""
		Image Preprocessing steps, to match image 
		with Input Neural nets
		Image,
		N, Channel, Height, Width
		"""
		blob = cv2.resize(image, (w, h))  # Resize width & height
		blob = blob.transpose((2, 0, 1))  # Change data layout from HWC to CHW
		blob = blob.reshape((n, c, h, w))
		return blob


	def processFace(self, image):

		blob = self.image_preprocessing(
			image, self.n_facedetect, self.c_facedetect, self.h_facedetect, self.w_facedetect)

		res = self.exec_net.infer(inputs={self.FACEDETECT_INPUTKEYS: blob})
		res = res[self.FACEDETECT_OUTPUTKEYS]

		# Get Bounding Box Result
		for detection in res[0][0]:
			confidence = float(detection[2])  # Face detection Confidence
			# Obtain Bounding box coordinate, +-10 just for padding
			xmin = int(detection[3] * image.shape[1] - 10)
			ymin = int(detection[4] * image.shape[0] - 10)
			xmax = int(detection[5] * image.shape[1] + 10)
			ymax = int(detection[6] * image.shape[0] + 10)

			fontColor = (0, 0, 255)

			# Crop Face which having confidence > 90%
			if confidence > 0.6:
				# Draw Boundingbox
				cv2.rectangle(image, (xmin, ymin), (xmax, ymax), fontColor)
			
		return image


	def processFaceRectangles(self, image):

		blob = self.image_preprocessing(
			image, self.n_facedetect, self.c_facedetect, self.h_facedetect, self.w_facedetect)

		res = self.exec_net.infer(inputs={self.FACEDETECT_INPUTKEYS: blob})
		res = res[self.FACEDETECT_OUTPUTKEYS]

		return res[0][0]


	def rectangleArrayDrawOnImage(self, res, image):

		# Get Bounding Box Result
		for detection in res:
			confidence = float(detection[2])  # Face detection Confidence
			# Obtain Bounding box coordinate, +-10 just for padding
			xmin = int(detection[3] * image.shape[1] - 10)
			ymin = int(detection[4] * image.shape[0] - 10)
			xmax = int(detection[5] * image.shape[1] + 10)
			ymax = int(detection[6] * image.shape[0] + 10)

			fontColor = (0, 0, 255)

			# Crop Face which having confidence > 90%
			if confidence > 0.6:
				# Draw Boundingbox
				cv2.rectangle(image, (xmin, ymin), (xmax, ymax), fontColor)
			
		return image



FACEDETECT_XML = "models/face-detection-adas-0001.xml"
FACEDETECT_BIN = "models/face-detection-adas-0001.bin"


faceDetector=FaceDeteClass(FACEDETECT_XML, FACEDETECT_BIN)


if __name__ == '__main__':

	frame = cv2.imread('nomask_45.jpg')

	faceRectanglesArray = faceDetector.processFaceRectangles(frame)

	faceFrame = faceDetector.rectangleArrayDrawOnImage(faceRectanglesArray, frame)

	cv2.imshow('ssss', faceFrame)

	key = cv2.waitKey(0)