import numpy as np
import re
import sys

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(file, remove_inf = True):
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().decode('utf-8').rstrip()
  if header == 'PF':
    color = True
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')
  dim_line = file.readline().decode('utf-8').strip()
  dims_found= dim_line.split(" ")
  width, height = map(int, dims_found)

  scale_line = file.readline().decode('utf-8').rstrip()
  scale = float(scale_line)
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  img =  np.flip(np.reshape(data, shape), axis=0)
  if(remove_inf):
      img = np.where(img==np.inf, 0, img)
  return img, scale

'''
Save a Numpy array to a PFM file.
'''
def save_pfm(file, image, scale = 1):
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)

if __name__ =="__main__":
    p = "D:\gdrive\python_projects\FYP_FINAL\\benchmarking\MiddEval\MiddEval3\\trainingQ\Adirondack\disp0GT.pfm"
    f = open(p, 'rb')
    img = load_pfm(f)
    import matplotlib.pyplot as plt
    plt.imshow(img[0])

