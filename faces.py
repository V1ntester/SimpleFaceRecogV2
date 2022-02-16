import os, cv2, numpy, dlib

from skimage import io
from scipy.spatial import distance

class Faces():
	@staticmethod
	def GetFaces(shapePredic, faceRecog, detector):
		descPersons = {}
		names = []

		dirImages = os.listdir('faceBase')
		for i in range(0, len(dirImages)):
			try:
				name, _ = map(str, dirImages[i].split('.'))
				frame = io.imread(f'faceBase/{str(dirImages[i])}')

				faceDescFrame = Faces.GetFaceDesc(frame, shapePredic, faceRecog, detector)

				if not faceDescFrame:
					continue

				names.append(name)
				descPersons[name] = faceDescFrame
			except Exception as e:
				print(e)

		return [names, descPersons]

	@staticmethod
	def FindFaces(frame, data, shapePredic, faceRecog, detector):
		names = data[0]
		descPersons = data[1]

		minEuc = 1
		matchedName = ''

		faceDescFrame = Faces.GetFaceDesc(frame, shapePredic, faceRecog, detector)

		if not faceDescFrame:
			pass
		else:
			for i in range(0, len(names)):
				a = distance.euclidean(faceDescFrame, descPersons[names[i]])

				if a < minEuc:
					minEuc = a
					matchedName = names[i]

		return [minEuc, matchedName]

	def GetFaceDesc(frame, shapePredic, faceRecog, detector):
		detsShape = detector(frame, 1)

		for k, d in enumerate(detsShape):
			shapeFrame = shapePredic(frame, d)

		if 'shapeFrame' in locals():
			faceDescFrame = faceRecog.compute_face_descriptor(frame, shapeFrame)
			return faceDescFrame
		else:
			return False

