"""
PIL 패키지와 numpy 패키지를 이용하면,
이미지 파일(jpg, png, bmp, ...)의 픽셀 정보를 numpy.ndarray 형식으로 변환하거나
numpy.ndarray 형식의 이미지 픽셀 정보를 이미지 파일로 저장할 수 있습니다.
"""
from PIL import Image
import numpy as np

def image_to_pixel(image_file):
    """이미지 파일 이름(경로)를 파라미터로 전달받아서,
    numpy.ndarray에 픽셀 정보를 저장해서 리턴."""
    img = Image.open(image_file,mode='r') #  이미지 파일 오픈
    print(type(img))
    pixels = np.array(img)
    print('pixels shape', pixels.shape)
    #color : 8bit(gray scale), 24bit(rgb), 32bit(rgba -> rgb+ 불투명도)
    return pixels

def pixel_to_image(pixel, image_file):
    """numpy.ndarray 형식의 이미지 픽셀 정보와, 저장할 파일 이름을 파라미터로
    전달받아서, 이미지 파일을 저장"""
    img = Image.fromarray(pixel) #ndarray 타입의 데이터를 이미지로 변환
    print(type(img))
    img.show()
    img.save(image_file) #이미지 객체로 저장

if __name__ == '__main__':
    pixels_1 = image_to_pixel("chrismas.jpg")
    pixel_to_image(pixels_1,'test1.jpg')




