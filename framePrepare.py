import cv2, numpy

class FramePrepare():
	@staticmethod
	def CompressFrame(frame, scalePercent):
		width = int(frame.shape[1] * scalePercent / 100)
		height = int(frame.shape[0] * scalePercent / 100)

		dim = (width, height)

		compressedFrame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
		return compressedFrame