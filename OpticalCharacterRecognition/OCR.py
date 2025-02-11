import easyocr
reader = easyocr.Reader(['en'])
result = reader.readtext('OpticalCharacterRecognition/testImages/frame1.jpg')
print(len(result) > 0)