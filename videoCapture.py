import cv2, numpy

class VideoCapture():

	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

	@classmethod
	def ReleaseCapture(self):
		self.cap.release()

	@classmethod
	def ReadFrame(self):
		ret, frame = self.cap.read()
		return frame
