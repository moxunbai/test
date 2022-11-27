
import taichi as ti
import numpy as np

import freetype 


ti.init(arch=ti.gpu)


# @ti.kernel
# def run(n:ti.i32):
#     for i in range(n):
#         print(i)


# run(4) 

win_w=600
win_h=600
#B = (1-t)*P0+t*P1
def one_bezier_curve(a, b, t):
    return (1-t)*a + t*b

#使用de Casteljau算法求解曲线
def n_bezier_curve(x, n, k, t):
    #当且仅当为一阶时，递归结束
    if n == 1:
        return one_bezier_curve(x[k], x[k+1], t)
    else:
        return (1-t)*n_bezier_curve(x, n-1, k, t) + t*n_bezier_curve(x, n-1, k+1, t)
 
def bezier_curve(x, y, num, b_x, b_y):
    #n表示阶数
    n = len(x) - 1
    t_step = 1.0 / (num - 1)
    t = np.arange(0.0, 1+t_step, t_step)
    for each in t:
        b_x.append(n_bezier_curve(x, n, 0, each))
        b_y.append(n_bezier_curve(y, n, 0, each))

x=[0,4,9]
y=[0,8,2]
b_x = []
b_y = []

# bezier_curve(x,y,18,b_x,b_y)
# print(b_x)
# print(b_y)

class put_chinese_text(object):
 def __init__(self, ttf):
    self._face = freetype.Face(ttf)

 def draw_text(self,  pos, text, text_size, text_color):
        '''
        draw chinese(or not) text with ttf
        :param image:  image(numpy.ndarray) to draw text
        :param pos:  where to draw text
        :param text:  the context, for chinese should be unicode type
        :param text_size: text size
        :param text_color:text color
        :return:   image
        '''
        self.text_size=text_size
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender / 64.0

        # descender = metrics.descender/64.0
        # height = metrics.height/64.0
        # linegap = height - ascender + descender
        ypos = int(ascender)

        text = text
        img = self.draw_string( pos[0], pos[1] + ypos, text, text_color)
        return img

 def draw_string(self,   x_pos, y_pos, text, color):
  '''
  draw string
  :param x_pos: text x-postion on img
  :param y_pos: text y-postion on img
  :param text: text (unicode)
  :param color: text color
  :return:  image
  '''
  prev_char = 0
  pen = freetype.Vector()
  pen.x = x_pos << 6 # div 64
  pen.y = y_pos << 6

  hscale = 1.0
  matrix = freetype.Matrix(int(hscale) * 0x10000, int(0.2 * 0x10000), \
         int(0.0 * 0x10000), int(1.1 * 0x10000))
  cur_pen = freetype.Vector()
  pen_translate = freetype.Vector()
  
  img_list = []
#   image = copy.deepcopy(img)
  for cur_char in text:
   self._face.set_transform(matrix, pen_translate)

   self._face.load_char(cur_char)
   kerning = self._face.get_kerning(prev_char, cur_char)
   pen.x += kerning.x
   slot = self._face.glyph
   bitmap = slot.bitmap

   cur_pen.x = pen.x
   cur_pen.y = pen.y - slot.bitmap_top * 64
   img_list.append(self.draw_ft_bitmap(  bitmap, cur_pen, color))
   

   pen.x += slot.advance.x
   prev_char = cur_char

  return img_list

 def draw_ft_bitmap(self,   bitmap, pen, color):
  '''
  draw each char
  :param bitmap: bitmap
  :param pen: pen
  :param color: pen color e.g.(0,0,255) - red
  :return:  image
  '''
  x_pos = pen.x>>6
  y_pos = pen.y>>6
  cols = bitmap.width
  rows = bitmap.rows
  img=np.zeros(shape=(text_size+2,text_size+2,3))
  glyph_pixels = bitmap.buffer
  
  offset_x=0
  offset_y=0
  if text_size>rows:
      offset_y = int((text_size-rows)/2)
  if text_size>cols:
      offset_x = int((text_size-cols)/2)

  for row in range(rows):
   for col in range(cols):
    if glyph_pixels[row * cols + col] != 0:
     try: 
      imx_x= offset_x+col
      imx_y=rows- row  -offset_y
      img[imx_x][imx_y][0] = color[0]
      img[imx_x][imx_y][1] = color[1]
      img[imx_x][imx_y][2] = color[2]
     except:
      continue
  return img


line = '毛不易12asd'
frame_img = np.zeros(shape=(win_w,win_h,3))

color_ = (0, 255, 0) # Green
pos = (3, 3)
text_size = 14
ft = put_chinese_text('Alimama_ShuHeiTi_Bold.ttf')
images = ft.draw_text( pos, line, text_size, color_)

start_pos_x=0
for img in images:
    print(img.shape)
    frame_img[start_pos_x:start_pos_x+img.shape[0],win_h-img.shape[1]:win_h,:]=img
    start_pos_x+=img.shape[0]

 
gui = ti.GUI('Hello World!', (win_w,win_h)) 
while gui.running:
    gui.set_image(frame_img) 
    gui.show() 