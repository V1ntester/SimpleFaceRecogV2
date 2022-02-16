import cv2, numpy, dlib

from videoCapture import VideoCapture
from framePrepare import FramePrepare
from faces import Faces

class Main():
	videoCapture = VideoCapture()

	shapePredic = dlib.shape_predictor('dlib/shape_predictor_68_face_landmarks.dat')
	faceRecog = dlib.face_recognition_model_v1('dlib/dlib_face_recognition_resnet_model_v1.dat')
	detector = dlib.get_frontal_face_detector()

	def __init__(self):
		data = Faces.GetFaces(self.shapePredic, self.faceRecog, self.detector)

		cv2.startWindowThread()
		cv2.namedWindow("preview")

		while(True): 
			frame = self.videoCapture.ReadFrame()

			minEuc, matchedName = map(str, Faces.FindFaces(FramePrepare.CompressFrame(frame, 30), data, self.shapePredic, self.faceRecog, self.detector))

			minEuc = float(minEuc)

			if minEuc < 0.6:
				cv2.putText(frame, f'Person: {matchedName}', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
			elif minEuc != 1:
				cv2.putText(frame, 'Person not identified', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)

			cv2.imshow('preview', frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				self.videoCapture.ReleaseCapture()


Main()