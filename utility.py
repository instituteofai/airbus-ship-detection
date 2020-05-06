import numpy as np

# Reference: https://www.kaggle.com/kmader/baseline-u-net-model-part-1
def rle_encode(img):
  """
  Get an RLE representation of a 2D mask

  Parameters:
    img: 2d numpy array with: 1 = mask, 0 = background

  Returns:
    string: RLE representation of mask

  """
  pixels = img.T.flatten()
  pixels = np.concatenate([[0], pixels, [0]])
  runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
  runs[1::2] -= runs[::2]
  return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
  """
  Decode an RLE representation into a 2D mask

  Parameters:
    mask_rle (string): String in RLE format
    shape (height, width): Shape of array to return

  Returns:
    2d numpy array with: 1 = mask, 0 = background
  """
  s = mask_rle.split()
  starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
  starts -= 1
  ends = starts + lengths
  img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
  for lo, hi in zip(starts, ends):
      img[lo:hi] = 1
  return img.reshape(shape[1], shape[0]).T
  