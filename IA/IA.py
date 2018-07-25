from __future__ import print_function
from tkinter import filedialog
from tkinter import *

from PIL import Image, ImageDraw

import os
import statistics
import math

import csv
import warnings

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'toolmanager'
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolZoom

warnings.filterwarnings("ignore")


LEFT = "Left"
RIGHT = "Right"
TOP = "Top"
BOTTOM = "Bottom"


def min_position(row):
    smallest = 255
    for i in row:
        if i[1] < smallest:
            smallest = i[1]
            position = i[0]
    return position


def find_2nd_derivative(x, y):
    left = 1
    position = 0
    x_max = 0
    y_max = 0
    max_position = 0
    cut_off = 0.85
    for i in range(len(x)):
        if y[i] > y_max:
            max_position, y_max = x[i], y[i]
            x_max = i

    if y[0] > y[-1]:
        left = 0
    elif y[0] == y[-1]:
        if (max_position-x[0]) > (x[-1]-max_position):
            left = 0

    if left == 1:
        if y[-1] >= cut_off * y_max:
            position = x[-1]
        else:
            for i in range(x_max, len(x)-1):
                if y[i] >= cut_off * y_max >= y[i+1]:
                    position = x[i]
    else:
        if y[0] >= cut_off * y_max:
            position = x[0]
        else:
            for i in range(0, x_max):
                if y[i] <= cut_off * y_max <= y[i+1]:
                    position = x[i]
    return left, position


def correlate(l, region):
    n, j = 0, 0
    longest = 0
    left, right = region[0], region[1]
    for i in l:
        region_list = [i[0], i[1], left, right]
        region_list.sort()
        if region_list[1] >= left and region_list[2] <= right:
            length = region_list[2]-region_list[1]
        else:
            length = 0

        if length > longest:
            longest = length
            n = j
        j = j + 1
    return n


def get_pixels(low, high, height, width, data):
    pixel_x = []
    pixel_y = []
    for i in range(height):
        for j in range(width):
            if data[i * width + j][0] in range(low, high):
                pixel_x.append(j)
                pixel_y.append(i)
    return pixel_x, pixel_y


def rect_boundary(t, l, b, r, a=0):
    rect_x = []
    rect_y = []
    for i in range(t, b + 1):
        left = l + int(round(math.tan(a) * (i - t)))
        right = r + int(round(math.tan(a) * (i - t)))
        rect_x.append(left)
        rect_y.append(i)
        rect_x.append(right)
        rect_y.append(i)
    for j in range(l, r + 1):
        rect_x.append(j)
        rect_y.append(t)
        rect_x.append(j + int(round((b - t) * math.tan(a))))
        rect_y.append(b)

    return rect_x, rect_y


def find_white(l, white=254):
    count = 0
    white_line = 0
    position = (0, 0)
    for i in range(len(l)):
        if l[i] == white:
            count += 1
        else:
            if count > white_line:
                white_line = count
                position = (i - count, i)
            count = 0
    return white_line, position


def find_bar_region(data, w, h):
    top = 0
    bottom = h
    left = 0
    right = w
    for i in range(880, h):
        num = 0
        for j in range(w):
            stride = j + w * i
            if data[stride][0] <= 60:
                num += 1
        if num > 0.95 * w:
            top = i
            break
    j = w - 20
    while j >= 0:
        j = j - 1
        num = 0
        for i in range(top, bottom):
            stride = j + w * i
            if data[stride][0] >= 100:
                num += 1
        if num > 0.95 * (bottom - top):
            left = j
            break
    for i in range(top + 10, h):
        num = 0
        for j in range(left, right):
            stride = j + w * i
            if data[stride][0] >= 100:
                num += 1
        if num > 0.95 * (right - left):
            bottom = i
            break
    
    return top, bottom, left, right


def line2scatter(line_x, line_y, div=1):
    x_list = [div*int(round(i)) for i in line_x]
    y_list = [div*int(round(i)) for i in line_y]
    x_scatter = []
    y_scatter = []
    for j in range(len(line_x) - 1):
        if x_list[j + 1] > x_list[j]:
            for x in range(x_list[j], x_list[j + 1]):
                y = y_list[j] + (y_list[j + 1] - y_list[j]) / (x_list[j + 1] - x_list[j]) * (x - x_list[j])
                x_scatter.append(x)
                y_scatter.append(round(y))

        elif x_list[j + 1] == x_list[j]:
            x = x_list[j]
            for y in range(y_list[j], y_list[j + 1]):
                x_scatter.append(x)
                y_scatter.append(y)
        else:
            for i in range(x_list[j] - x_list[j + 1]):
                x = x_list[j] - i
                y = y_list[j] + (y_list[j + 1] - y_list[j]) / (x_list[j + 1] - x_list[j]) * (x - x_list[j])
                x_scatter.append(x)
                y_scatter.append(round(y))

    if div != 1:
        x_scatter = [i/div for i in x_scatter]
        y_scatter = [j/div for j in y_scatter]
    return x_scatter, y_scatter


def in_out(x, y, x_scatter, y_scatter):
    pair = [-1, -1]
    for i in range(len(x_scatter)):
        if x_scatter[i] == x:
            if pair[0] == -1:
                pair[0] = y_scatter[i]
            else:
                pair[1] = y_scatter[i]
                if pair[0] <= y <= pair[1]:
                    return True
                if pair[1] <= y <= pair[0]:
                    return True
                pair = [-1, -1]
    return False


class ModifiedZoom(ToolZoom):
    def __init__(self, *args):
        ToolZoom.__init__(self, *args)

    def enable(self, event):
        """Connect press/release events and lock the canvas"""
        self.figure.canvas.widgetlock(self)
        self._idPress = self.figure.canvas.mpl_connect(
            'button_press_event', self._press)
        self._idRelease = self.figure.canvas.mpl_connect(
            'button_release_event', self._release)
        self._idScroll = self.figure.canvas.mpl_connect(
            'scroll_event', self.scroll_zoom)
        global connect_flag
        connect_flag = True

    def disable(self, event):
        """Release the canvas and disconnect press/release events"""
        self._cancel_action()
        self.figure.canvas.widgetlock.release(self)
        self.figure.canvas.mpl_disconnect(self._idPress)
        self.figure.canvas.mpl_disconnect(self._idRelease)
        self.figure.canvas.mpl_disconnect(self._idScroll)

        global connect_flag
        connect_flag = False
        global my_ui
        if my_ui:
            my_ui.threshold_scatter()

    def scroll_zoom(self, event):
        self.figure.canvas.draw_idle()
        return


class Parameters:
    def __init__(self):
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None
        self.angle = 0
        self.field_position = 0
        self.discontinuity = 0
        self.noise_level = 2

        self.threshold = 0
        self.threshold_list_x = []
        self.threshold_list_y = []

        self.gray_threshold = 125
        self.gray_threshold_list_x = []
        self.gray_threshold_list_y = []

        self.black_x = []
        self.black_y = []
        self.gray_x = []
        self.gray_y = []
        self.white_x = []
        self.white_y = []

        self.grid_parameter = 50
        self.grid_line_list = []
        self.v_grid = []
        self.h_grid = []

# 如果ready 就可以在 pyplot 的地方画图了
    def is_ready(self):
        return self.left and self.right and self.top and self.bottom and self.threshold

    def set_boundary_values(self, left, right, top, bottom):
        if left and right and top and bottom:
            self.left = round(float(left))
            self.right = round(float(right))
            self.top = round(float(top))
            self.bottom = round(float(bottom))
        else:
            self.left = None
            self.right = None
            self.top = None
            self.bottom = None

    def set_angle_value(self, angle):
        self.angle = float(angle) / 180 * math.pi

# 和下面的get edge 一起找到单一threshold 下的边界
    def set_threshold_list(self, height, width, data, r=True):
        self.threshold_list_x = []
        self.threshold_list_y = []
        self.threshold_list_x, self.threshold_list_y = self.get_edge(self.threshold, height, width, data, region=r)

### 根据threshold 找边界 并返回一个边界的list 用于作图。 region确定是不是找整张图的边界，还是zoom in之后的边界。
    def get_edge(self, threshold, height, width, data, region=True):
        assert height
        assert width
        assert data
        threshold_x = []
        threshold_y = []
        if region:
            height_region = (0, height)
            width_region = (0, width)
        else:
            height_region = (self.top, self.bottom)
            width_region = (self.left, self.right)
        for i in range(*height_region):
            for j in range(*width_region):
                stride = (i * width) + j
                last_line = (i - 1) * width + j
                next_line = (i + 1) * width + j
                if data[stride][0] <= threshold:
                    if i == 0 or i == (height - 1):
                        threshold_x.append(i)
                        threshold_y.append(j)
                    else:
                        if max(data[last_line][0], data[next_line][0], data[stride - 1][0],
                               data[stride + 1][0]) > threshold:
                            threshold_x.append(i)
                            threshold_y.append(j)
        return threshold_x, threshold_y

# 给 threshold 赋值
    def set_threshold_value(self, threshold):
        self.threshold = threshold

# 给第二threshold 赋值 （其实这个也可以找到除了黑白以外第三个颜色）
    def set_gray_threshold_list(self, height, width, data):
        self.gray_threshold_list_x = []
        self.gray_threshold_list_y = []
        self.gray_threshold_list_x, self.gray_threshold_list_y = \
            self.get_edge(self.gray_threshold, height, width, data)

# 给gray threshold 赋值。
    def set_gray_threshold_value(self, gray_threshold):
        self.gray_threshold = gray_threshold

# 把灰色的区域找出来染色。
    def set_pixels(self, height, width, data):
        self.black_x, self.black_y = get_pixels(0, self.threshold + 1, height, width, data)
        self.gray_x, self.gray_y = get_pixels(self.threshold + 1, self.gray_threshold + 1, height, width, data)
        self.white_x, self.white_y = get_pixels(self.gray_threshold + 1, 256, height, width, data)

# 看看某个list是不是都是白的。好像跟外面的静态函数重复了....
    def all_white(self, l, start, end, direction=1, white=0):
        if not white:
            assert isinstance(self.grid_parameter, int)
            white = self.grid_parameter
        black = 0
        portion = 0
        if direction == 0:
            portion = 0
        for i in range(start, end):
            if l[i][0] < white:
                black += 1
            if black > portion * (end - start):
                return False
        return True

    def image_grid(self, height, width, data):

        grid_const_1 = 0
        grid_const_2 = 10
        grid_const_3 = 15

        im_v_grid = []
        im_h_grid = []
        v_grid = [-1, -1]

        for i in range(width):
            vertical_list = [data[i + width * j] for j in range(height)]
            if not self.all_white(vertical_list, self.field_position, len(vertical_list) - grid_const_1, direction=0) and v_grid[0] == -1:
                v_grid[0] = i
                v_grid[1] = 1
            if self.all_white(vertical_list, self.field_position, len(vertical_list) - grid_const_1, direction=0) and v_grid[1] == 1:
                v_grid[1] = i
                if v_grid[1] - v_grid[0] > grid_const_2:
                    im_v_grid.append(v_grid)
                v_grid = [-1, -1]
            if i == width - 1 and v_grid[1] == 1:
                v_grid[1] = i
                if v_grid[1] - v_grid[0] > grid_const_2:
                    im_v_grid.append(v_grid)
                v_grid = [-1, -1]
        self.v_grid = im_v_grid
        if len(im_v_grid):
            im_h_grid = [0] * len(im_v_grid)
        n = 0
        for k in im_v_grid:
            im_h_grid[n] = []
            im_h_grid[n] = self.get_grid((k[0], k[1]), width, height, data, grid_const_3)
            if len(im_h_grid[n]) <= self.noise_level:
                im_h_grid[n] = self.special_grid(k, width, height, data)
            n += 1
        self.h_grid = im_h_grid

        self.h_grid, self.v_grid = [], []
        length = len(im_v_grid)
        for i in range(length):
            if len(im_h_grid[i]) > self.noise_level:
                self.v_grid.append(im_v_grid[i])
                self.h_grid.append(im_h_grid[i])

        self.get_grid_line(height)

    def get_grid(self, region, width, height, data, grid_const):
        grids = [-1, -1]
        im_grid = []
        for j in range(height):
            row = [data[i + width * j] for i in range(*region)]
            if not self.all_white(row, 0, len(row)) and grids[0] == -1:
                grids[0] = j
                grids[1] = 1
            if self.all_white(row, 0, len(row)) and grids[1] == 1:
                grids[1] = j
                if grids[1] - grids[0] > grid_const:
                    im_grid.append(grids)

                    if len(im_grid) > 1:
                        if im_grid[-1][0] - im_grid[-2][1] <= self.discontinuity:
                            grids = [im_grid[-2][0], im_grid[-1][1]]
                            im_grid.pop()
                            im_grid.pop()
                            im_grid.append(grids)

                grids = [-1, -1]
            if j == height - 1 and grids[1] == 1:
                grids[1] = j
                if grids[1] - grids[0] > grid_const:
                    im_grid.append(grids)
                grids = [-1, -1]
        return im_grid

    def special_grid(self, region, width, height, data):
        mid = 0
        if region:
            mid = round((region[0]+region[1]) / 2)
        if self.left_right(mid, region, width, height, data):
            region = [mid, region[1]]
        else:
            region = [region[0], mid]
        return self.grid(width, height, region, data)

    def left_right(self, mid, region, width, height, data):
        column = [(j, data[j*width+mid][0]) for j in range(height)]
        white_range = [-2, -1]
        white_list = []
        longest = 0
        position = [0, 0]
        for i in column:
            if i[1] <= self.grid_parameter and white_range[0] == -2:
                white_range[0] = -1
            if i[1] > self.grid_parameter and white_range[0] == -1:
                white_range[0] = i[0]
            if i[1] <= self.grid_parameter and white_range[0] >= 0:
                white_range[1] = i[0]
                white_list.append(white_range)
                white_range = [-1, -1]

        for i in white_list:
            if i[1]-i[0] > longest:
                longest = i[1] - i[0]
                position = i
        middle = int(round((position[0]+position[1])/2))

        row = [(i, data[middle*width+i][0]) for i in range(*region)]
        if min_position(row) < mid:
            return True
        else:
            return False

        # longest = 0
        # position = 0
        # x_list, y_list = [], []
        # for i in range(*region):
        #     black = 0
        #     for j in range(height):
        #         if data[j * width + i][0] <= self.grid_parameter:
        #             black += 1
        #     x_list.append(i)
        #     y_list.append(black)
        #     if black > longest:
        #         longest = black
        #         position = i
        # if position > mid:
        #     return False
        # return True

    def grid(self, width, height, region, data):
        grid_const_3 = 15
        return self.get_grid(region, width, height, data, grid_const_3)

    def get_grid_line(self, height):
        self.grid_line_list = []
        n = 0
        for k in self.v_grid:
            for l in range(2):
                line_x = [k[l]]*height
                line_y = [i for i in range(height)]
                self.grid_line_list.append([line_x, line_y])

            for m in self.h_grid[n]:
                for l in range(2):
                    line_x = [i for i in range(k[0], k[1])]
                    line_y = [m[l]] * (k[1]-k[0])
                    self.grid_line_list.append([line_x, line_y])
            n += 1


class TEMImage:
    def __init__(self):
        self.image = ""
        self.data_tuple = ""
        self.data = []
        self.original_data = []
        self.width = 0
        self.height = 0
        self.image_path = ""
        self.image_type = "STEM"
        self.filename = ''

        self.im_v_grid = []
        self.im_h_grid = []
        self.grid_list = []

        self.scale_bar = 50
        self.length = 250
        self.scale_x = []
        self.scale_y = []
        self.ratio = 0.2

        self.section_list = []
        self.column_list = []
        self.thickness_matrix = []
        self.black_pixels = []
        self.thickness_list = []
        self.thickness_column = []
        self.thickness = 0
        self.standard_deviation = 0

        self.white_pixels = []
        self.void_list = []
        self.void_column = []
        self.void_matrix = []
        self.void_percentage = 0
        self.void_sd = 0

        self.vp_sd = 0

        self.sidewall = 0
        self.roughness = 0
        self.sw_list = []
        self.sw_rough_list = []
        self.sw_thick_list = []
        self.sw_vertical = []
        self.sw_section_list = []

        self.void_xs_list = []
        self.void_ys_list = []
        self.sd_position = ()

    def open_file(self, file_path):
        assert os.path.exists(file_path)
        self.image = Image.open(file_path).convert("RGB")
        self.width, self.height = self.image.size
        self.data_tuple = self.image.getdata()
        self.data = [i for i in self.data_tuple]
        self.original_data = self.data
        file_name = os.path.split(file_path)[1]
        self.filename = file_name
        if 'OUTS' in file_name:
            self.image_type = 'TEM'
        if 'OUTS' not in file_name:
            self.image_type = 'STEM'

        scale_dict = {'-a': 1000, '-b': 200, '-c': 50, '20K': 2000, '20k': 2000, '80K': 500, '80k': 500, '250K': 200,
                      '250k': 200}
        for key in scale_dict.keys():
            if key in file_name:
                self.scale_bar = scale_dict[key]
                break
        self.convert_ratio()
        self.void_xs_list = []
        self.void_ys_list = []

    def convert_ratio(self):
        if self.image_type == 'TEM':
            y = 0
            length = 0
            index = ()
            for j in range(self.height - 100, self.height):
                line = [self.data[j * self.width + i][0] for i in range(400)]
                length, index = find_white(line)
                if length > 50:
                    y = j
                    break
            self.scale_x = [i for i in range(*index)]
            self.scale_y = [y + 10 for i in range(*index)]
            try:
                self.length = length
                self.ratio = self.scale_bar / length
            except ZeroDivisionError as e:
                print('Scale bar error', e)
        if self.image_type == 'STEM':
            start, end = 0, 0
            t, b, l, r = find_bar_region(self.data, self.width, self.height)

            m = round(0.5 * t + 0.5 * b)
            for j in range(l + 3, r - 3):
                stride = j + m * self.width
                if self.data[stride][0] >= 250:
                    start = j
                    break
            for j in range(l + 3, r - 3):
                stride = j + m * self.width
                if self.data[stride][0] >= 250:
                    end = j
            self.scale_x = [i for i in range(start, end)]
            self.scale_y = [m + 10 for i in range(start, end)]
            try:
                self.length = end - start
                self.ratio = self.scale_bar / self.length
            except ZeroDivisionError as e:
                print('Scale bar error', e)
            finally:
                self.ratio = 1
                self.length = 200

            self.height = t

        return

    def update_ratio(self, new_bar):
        self.scale_bar = new_bar
        self.ratio = self.scale_bar / self.length
        return

    def get_grid(self, parameters):
        assert isinstance(parameters, Parameters)
        parameters.image_grid(self.height, self.width, self.data)
        self.im_v_grid = parameters.v_grid
        self.im_h_grid = parameters.h_grid
        self.grid_list = parameters.grid_line_list

    def all_white(self, row, white):
        for i in row:
            if i < white:
                return False
        return True

    def pixels(self, region, parameters, position=False):
        l, r, t, b = region
        pixels_list = []
        x_list, y_list = [], []
        for j in range(l, r):
            black = 0
            for i in range(t, b):
                stride = j + i * self.width
                if self.data[stride][0] < parameters.threshold:
                    black += 1
            pixels_list.append(black * math.cos(parameters.angle) * self.ratio)
            x_list.append(j)
            y_list.append(black)
        if position:
            return x_list, y_list
        return pixels_list

##### 5(c) 计算sidewall roughness。 roughness的定义是standard deviation。 之后有可能会改。
    # 也可以算水平区域的厚度和roughness
    def calculate_sidewall(self, parameters, horizontal):
        l, r, t, b, a = parameters.left, parameters.right, parameters.top, parameters.bottom, parameters.angle

        self.sw_list = []
        if horizontal == 0:
            plot_list_x, plot_list_y = self.pixels((l, r, t, b), parameters, position=True)
            l_r, position = find_2nd_derivative(plot_list_x, plot_list_y)
            if l_r == 1:
                r = position
                self.sd_position = ([int(round(r + (i - t) * math.tan(a))) for i in range(t, b)], [i for i in range(t, b)])
            else:
                l = position
                self.sd_position = ([int(round(l + (i - t) * math.tan(a))) for i in range(t, b)], [i for i in range(t, b)])

            for i in range(t, b):
                black = 0
                left = int(round(l + (i - t) * math.tan(a)))
                right = int(round(r + (i - t) * math.tan(a)))
                for j in range(left, right):
                    stride = j + i * self.width
                    if self.data[stride][0] < parameters.threshold:
                        black += 1
                if black:
                    self.sw_list.append(black * math.cos(parameters.angle) * self.ratio)
        else:
            self.sw_list = self.pixels((l, r, t, b), parameters)

        self.sidewall = statistics.mean(self.sw_list)
        self.roughness = statistics.stdev(self.sw_list)

##### 5(c) 从image griding 继承来 sidewall的griding。 基本算法就是反选所有的wordline regio。
    def calculate_sidewall_section(self, parameters):
        self.sw_rough_list = []
        self.sw_thick_list = []
        self.sw_section_list = []
        l, r, t, b, a = parameters.left, parameters.right, parameters.top, parameters.bottom, parameters.angle
        sidewall_list = []
        n = correlate(self.im_v_grid, (parameters.left, parameters.right))
        for i in range(len(self.im_h_grid[n]) - 1):
            if self.im_h_grid[n][i][1] >= parameters.top and self.im_h_grid[n][i + 1][0] <= parameters.bottom:
                sidewall_list.append([self.im_h_grid[n][i][1], self.im_h_grid[n][i + 1][0]])
        for k in sidewall_list:
            sw_black_list = []
            for i in range(*k):
                black = 0
                left = int(round(l + (i - t) * math.tan(a)))
                right = int(round(r + (i - t) * math.tan(a)))
                for j in range(left, right):
                    stride = j + i * self.width
                    if self.data[stride][0] < parameters.threshold:
                        black += 1
                sw_black_list.append(black * math.cos(parameters.angle) * self.ratio)
            thickness = statistics.mean(sw_black_list)
            roughness = statistics.stdev(sw_black_list)
            self.sw_thick_list.append(thickness)
            self.sw_rough_list.append(roughness)
            self.sw_vertical.append([thickness, roughness])
        self.sw_section_list = sidewall_list

##### 5(a)（b) 请求计算厚度和void if void==true。 优先判断是不是有draw过region。如果有，把region涂白当作void。
### 从image grid继承来section list
### 把分析区域自动分成十个 column list
### 计算，还原涂白的数据。
    def calculate_thickness(self, parameters, voids=False):
        assert isinstance(parameters, Parameters)
        assert parameters.left
        assert parameters.right
        assert parameters.top
        assert parameters.bottom
        assert parameters.threshold

        self.draw_judgement(parameters)
        self.get_section_list(parameters)
        self.get_column_list(parameters)
        self.calculate(parameters, void=voids)
        self.data = self.original_data

    def draw_judgement(self, parameters):
        if self.void_xs_list:
            for i in range(parameters.top, parameters.bottom):
                for j in range(parameters.left, parameters.right):
                    for v in range(len(self.void_xs_list)):
                        if in_out(j, i, self.void_xs_list[v], self.void_ys_list[v]):
                            self.data[j+i*self.width] = (230, 230, 230)
        return

    def get_section_list(self, parameters):
        self.section_list = []
        n = correlate(self.im_v_grid, (parameters.left, parameters.right))
        for i in self.im_h_grid[n]:
            if i[0] >= parameters.top and i[1] <= parameters.bottom:
                self.section_list.append(i)

    def get_column_list(self, parameters):
        self.column_list = []
        r, l = parameters.right, parameters.left

        for j in range(l, r):
            column = [self.data[j + i * self.width][0] for i in range(parameters.top, parameters.bottom)]
            if not self.all_white(column, parameters.threshold):
                l = j
                break
        for j in range(l, r):
            column = [self.data[j + i * self.width][0] for i in range(parameters.top, parameters.bottom)]
            if not self.all_white(column, parameters.threshold):
                r = j

        parameters.right, parameters.left = r + 1, l
        column_width = round((r - l + 1) / 10)
        for i in range(9):
            self.column_list.append((l + i * column_width, l + (i + 1) * column_width))
        self.column_list.append((l + 9 * column_width, r + 1))
        print(self.column_list)

    def calculate(self, parameters, void=False):
        self.black_pixels = [[0 for x in range(10)] for y in range(len(self.section_list))]
        self.white_pixels = [[0 for x in range(10)] for y in range(len(self.section_list))]
        self.thickness_matrix = [[0 for x in range(10)] for y in range(len(self.section_list))]
        self.void_matrix = [[0 for x in range(10)] for y in range(len(self.section_list))]
        self.thickness_list = []
        self.thickness_column = []
        self.void_list = []
        self.void_column = []
        total_length = parameters.right - parameters.left

        for k in range(len(self.section_list)):
            for c in range(len(self.column_list)):
                black, white = 0, 0
                for i in range(self.column_list[c][0], self.column_list[c][1]):
                    start, end = 0, 0
                    for j in range(self.section_list[k][0], self.section_list[k][1]):
                        stride = i + self.width * j
                        if self.data[stride][0] < parameters.threshold:
                            start = j
                            break
                    for j in range(self.section_list[k][0], self.section_list[k][1]):
                        stride = i + self.width * j
                        if self.data[stride][0] < parameters.threshold:
                            end = j

                    for j in range(start, end+1):
                        stride = i + self.width * j
                        if self.data[stride][0] < parameters.threshold:
                            black += 1
                        else:
                            white += 1

                self.black_pixels[k][c] = black
                self.white_pixels[k][c] = white
                self.thickness_matrix[k][c] = (
                        self.ratio * black / (self.column_list[c][1] - self.column_list[c][0]) / 2)

                if void:
                    if white + black == 0:
                        voids = 0
                    else:
                        voids = white / (white + black)
                    self.void_matrix[k][c] = voids

            self.thickness_list.append(self.ratio * sum(self.black_pixels[k]) / total_length / 2)
            if void:
                self.void_list.append(
                    sum(self.white_pixels[k]) / (sum(self.white_pixels[k]) + sum(self.black_pixels[k])))

        if not self.thickness_list:
            print('Fail to grid, check the griding parameters')
            return
        if len(self.thickness_list) > 1:
            self.thickness = statistics.mean(self.thickness_list)
            self.standard_deviation = statistics.stdev(self.thickness_list)
        else:
            self.thickness = self.thickness_list[0]
            self.standard_deviation = 0
        self.thickness_column = [statistics.mean([self.thickness_matrix[i][j] for i in range(len(self.section_list))])
                                 for j in range(10)]

        if void:
            if len(self.void_list) > 1:
                self.void_percentage = statistics.mean(self.void_list)
                self.void_sd = statistics.stdev(self.void_list)
            else:
                self.void_percentage = self.void_list[0]
                self.standard_deviation = 0
            self.void_column = [statistics.mean([self.void_matrix[i][j] for i in range(len(self.section_list))])
                                for j in range(10)]


class UI:
    def __init__(self):
        self.root = Tk()
        self.temImage = TEMImage()
        self.parameters = Parameters()

        self.column_width = 20
        self.check = 0

        self.output_table = []
        self.sw_table = []

        self.lf_width, self.lf1_height = 400, 100
        self.lf2_width, self.lf2_height = 400, 100
        self.rf_width, self.rf_height = 400, 200
        self.btm_width, self.btm_height = 800, 200
        self.table_width, self.table_height = 800, 400
        self.sw_width, self.sw_height = 400, 400

        self.left_frame1 = LabelFrame(self.root, text='Image', relief=SUNKEN, bd=1,
                                      width=self.lf_width, height=self.lf1_height)
        self.left_frame2 = LabelFrame(self.root, text='Image Grid', relief=SUNKEN, bd=1,
                                      width=self.lf2_width, height=self.lf2_height)
        self.right_frame = LabelFrame(self.root, text='Analysis Region', relief=SUNKEN, bd=1, width=self.rf_width,
                                      height=self.rf_height)
        self.btm_frame = LabelFrame(self.root, text='Results', relief=SUNKEN, bd=1, width=self.btm_width,
                                    height=self.btm_height)
        self.table_frame = LabelFrame(self.root, text='Data', bd=0, width=self.table_width,
                                      height=self.table_height)
        self.sw_frame = LabelFrame(self.root, text='Sidewall Anaylsis', bd=0, width=self.sw_width,
                                   height=self.sw_height)

        self.threshold_ui = Entry(self.left_frame1)
        self.gray_threshold_ui = Entry(self.left_frame2)
        self.grid_parameter_ui = Entry(self.left_frame2)
        self.scale_bar_ui = Entry(self.left_frame1)
        self.mean_thick_ui = Entry(self.btm_frame)
        self.sd_thick_ui = Entry(self.btm_frame)
        self.mean_void_ui = Entry(self.btm_frame)
        self.sd_void_ui = Entry(self.btm_frame)
        self.sw_thickness_ui = Entry(self.btm_frame)
        self.sw_roughness_ui = Entry(self.btm_frame)
        self.boundary_ui = {}
        self.boundary_angle_ui = Entry(self.right_frame)

        self.var1 = IntVar()
        self.var2 = IntVar()
        self.var3 = IntVar()
        self.var4 = IntVar()

        self.xs = []
        self.ys = []
        self.draw_xs = []
        self.draw_ys = []
        self.line_list = []
        self.left_flag = 0

        self.scat = ''

# 打开STEM/TEM image的对话框，打开图片。 清空列表，override pyplot toolbar。初始化图片的对话框。
### 4. 插入convert ratio
    def open_file_ui(self):
        self.refresh()
        current_file = filedialog.askopenfilename(initialdir="C:/<whatever>", title="Select file",
                                                  filetypes=(("tif files", "*.tif"), ("all files", "*.*")))
        self.temImage.open_file(current_file)
        self.temImage.void_xs_list = []
        self.temImage.void_ys_list = []
        self.line_list = []
        self.insert_ratio_ui()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.mouse_wheel)
        self.fig.canvas.manager.toolmanager.add_tool('Zoom', ModifiedZoom)
        self.fig.canvas.manager.toolmanager.remove_tool('zoom')
        self.fig.canvas.manager.toolbar.add_tool('Zoom', 'navigation', 1)

        self.plot()

# 每次打开新图片，或者计算前初始化所有的数据。
    def refresh(self, t=True, b=True):
        table_height = len(self.output_table)
        if table_height:
            table_width = len(self.output_table[0])
            for i in range(table_height):
                for j in range(table_width):
                    if hasattr(self.output_table[i][j], 'destroy'):
                        self.output_table[i][j].destroy()

        sw_table_height = len(self.sw_table)
        if sw_table_height:
            sw_table_width = len(self.sw_table[0])
            for i in range(sw_table_height):
                for j in range(sw_table_width):
                    if hasattr(self.sw_table[i][j], 'destroy'):
                        self.sw_table[i][j].destroy()

        if b:
            labels = [LEFT, RIGHT, TOP, BOTTOM]
            for label_item in labels:
                self.boundary_ui[label_item].delete(0, 'end')
            self.parameters.set_boundary_values(None, None, None, None)
        if t:
            self.parameters.threshold = 0

### 4.和下面的两个function一起， 插入和更新convert ratio
    def insert_ratio_ui(self):
        if not self.var1.get():
            self.scale_bar_ui.delete(0, 'end')
            self.scale_bar_ui.insert(INSERT, self.temImage.scale_bar)
        self.update_ratio_ui()

    def update_ratio_ui(self):
        pad_x = 5
        pad_y = 2
        self.update_ratio()
        Label(self.left_frame1, text='Convert ratio: ').grid(sticky='W', row=2, column=0, padx=pad_x, pady=pad_y)
        Label(self.left_frame1, text=str('%.3f' % self.temImage.ratio)). \
            grid(sticky='W', row=2, column=1, padx=pad_x, pady=pad_y)
        return

    def update_ratio(self):
        update_bar = int(self.scale_bar_ui.get())
        if not update_bar or update_bar < 0:
            print("Invalid scale bar value")
            return
        self.temImage.update_ratio(update_bar)

##### 1. 3. drag box 的响应函数，用于点击左键。
    # 当var4 == 1 的时候开始draw。 当var4==0 的时候drag box。
    # 当使用右键的时候，封闭draw 的曲线。
    def on_press(self, event):

        if event.button == 1:
            if self.var4.get() == 1:
                if self.temImage.void_xs_list and self.left_flag == 0:
                    for v in range(len(self.temImage.void_xs_list)):
                        xs, ys = line2scatter(self.temImage.void_xs_list[v], self.temImage.void_ys_list[v], div=5)
                        self.ax.scatter(xs, ys, 0.1, color='yellow')
                self.left_flag = 1
                if event.inaxes != self.line.axes:
                    return
                self.draw_xs.append(event.xdata)
                self.draw_ys.append(event.ydata)

                self.line.set_data(self.draw_xs, self.draw_ys)
                self.line.figure.canvas.draw()
                return
            else:
                self.xs, self.ys = [], []
                if event.inaxes != self.line.axes:
                    return
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
        if event.button == 3:
            if self.var4.get() == 1 and self.left_flag == 1:
                if event.inaxes != self.line.axes:
                    return
                self.draw_xs.append(self.draw_xs[0])
                self.draw_ys.append(self.draw_ys[0])
                self.line.set_data(self.draw_xs, self.draw_ys)
                self.line.figure.canvas.draw()
                void_xs = self.draw_xs
                void_ys = self.draw_ys

                xs, ys = line2scatter(void_xs, void_ys)
                self.temImage.void_xs_list.append(xs)
                self.temImage.void_ys_list.append(ys)
                self.draw_xs, self.draw_ys = [], []
                self.left_flag = 0
                return

##### 3. delete button 用于删除draw的曲线的上一个点。 如果已经封闭了一组曲线，delete用于删除整个曲线。
    def delete_press(self, event):
        if event.key == 'delete':
            if self.var4.get() == 1 and self.left_flag == 1 and self.draw_xs:
                self.draw_xs.pop()
                self.draw_ys.pop()
                self.line.set_data(self.draw_xs, self.draw_ys)
                self.line.figure.canvas.draw()
            if self.var4.get() == 1 and self.left_flag == 0 and self.temImage.void_xs_list:
                self.temImage.void_xs_list.pop()
                self.temImage.void_ys_list.pop()
        return

### 3. 删除整个draw
    def clear_draw(self):
        self.temImage.void_xs_list = []
        self.temImage.void_ys_list = []
        self.plot(threshold_flag=True, boundary_flag=True)

    def on_motion(self, event):
        if not self.xs:
            return
        if event.inaxes != self.line.axes:
            return
        x0, y0 = self.xs[0], self.ys[0]
        x1, y1 = event.xdata, event.ydata

        rect_x = [x0, x0, x1, x1, x0]
        rect_y = [y0, y1, y1, y0, y0]

        self.line.set_data(rect_x, rect_y)
        self.line.figure.canvas.draw()

### 1. 松开鼠标是的响应函数 用于drag box的读取坐标。
    def on_release(self, event):

        if self.var4.get() == 1:
            return
        else:
            if event.inaxes != self.line.axes:
                return
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            rect_x = [self.xs[0], self.xs[0], self.xs[1], self.xs[1], self.xs[0]]
            rect_y = [self.ys[0], self.ys[1], self.ys[1], self.ys[0], self.ys[0]]
            self.bind_parameter()
            self.line.set_data(rect_x, rect_y)
            self.line.figure.canvas.draw()
            self.xs, self.ys = [], []

###### 将drag box的坐标值赋值给 parameters 并插入到ui 的boundary output
    def bind_parameter(self):
        labels = [LEFT, RIGHT, TOP, BOTTOM]
        for label_item in labels:
            self.boundary_ui[label_item].delete(0, 'end')
        self.parameters.set_boundary_values(None, None, None, None)

        boundary = ['%.1f' % self.xs[0], '%.1f' % self.xs[1], '%.1f' % self.ys[0], '%.1f' % self.ys[1]]
        self.parameters.set_boundary_values(*boundary)
        i = 0
        for label_item in labels:
            self.boundary_ui[label_item].insert(INSERT, boundary[i])
            i += 1

# 画threshold的散点图。
    def threshold_scatter(self, region=True):
        threshold = int(self.threshold_ui.get())
        self.parameters.set_threshold_value(threshold)
        self.parameters.set_threshold_list(self.temImage.height, self.temImage.width, self.temImage.data, r=region)
        if self.scat:
            self.scat.remove()
        if self.parameters.threshold_list_x and self.parameters.threshold_list_y:
            self.scat = self.ax.scatter(self.parameters.threshold_list_y, self.parameters.threshold_list_x, 0.05,
                                        color='yellow')

# 画图函数
##### 1. 4. 链接左键 ，右键， 以及delete button
    def plot(self, value=0, threshold_flag=False, gray_flag=False, grid=False, boundary_flag=False):
        plt.cla()
        self.ax.imshow(self.temImage.image)
        self.line, = self.ax.plot([0], [0], color='yellow')

        self.draw_xs = []
        self.draw_ys = []

        self.cid_press = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_delete = self.line.figure.canvas.mpl_connect('key_press_event', self.delete_press)
        self.cidmotion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        if grid:
            for i in self.temImage.grid_list:
                plt.scatter(i[0], i[1], 1, color='blue')
            plt.show()
            return

        if self.temImage.void_xs_list and self.temImage.void_ys_list:
            for v in range(len(self.temImage.void_xs_list)):
                xs, ys = line2scatter(self.temImage.void_xs_list[v], self.temImage.void_ys_list[v], div=5)
                self.ax.scatter(xs, ys, 0.1, color='yellow')

        if self.temImage.scale_x and self.temImage.scale_y:
            self.ax.scatter(self.temImage.scale_x, self.temImage.scale_y, 0.2, color='red')

        if self.parameters.threshold_list_x and self.parameters.threshold_list_y and threshold_flag:
            self.scat = self.ax.scatter(self.parameters.threshold_list_y, self.parameters.threshold_list_x, 0.05, color='yellow')

        if self.parameters.gray_threshold_list_x and self.parameters.gray_threshold_list_y and gray_flag:
            self.ax.scatter(self.parameters.gray_threshold_list_y, self.parameters.gray_threshold_list_x, 0.2,
                            color='red')

        if self.parameters.top and self.parameters.bottom and self.parameters.right and self.parameters.left and \
                boundary_flag:
            rect_x, rect_y = rect_boundary(self.parameters.top, self.parameters.left,
                                           self.parameters.bottom, self.parameters.right, self.parameters.angle)
            self.ax.scatter(rect_x, rect_y, 0.1, color='blue')

        if value == 1:
            for SectionText in range(0, len(self.temImage.section_list)):
                self.ax.text(self.parameters.left, self.temImage.section_list[SectionText][0],
                             'Layer ' + str(SectionText + 1) + ': ' +
                             str('%.2f' % self.temImage.thickness_list[SectionText]) + ' nm', fontsize=7)

        if value == 2:
            for SectionText in range(0, len(self.temImage.section_list)):
                self.ax.text((self.parameters.left + self.parameters.right) * 0.5,
                             self.temImage.section_list[SectionText][0],
                             'Layer ' + str(SectionText + 1) + ': ' +
                             str('%.2f' % self.temImage.void_list[SectionText]), fontsize=7)

        if value == 3:
            if self.var2.get() == 0:
                self.ax.scatter(*self.temImage.sd_position, 0.1, color='white')
                for SectionText in range(0, len(self.temImage.sw_thick_list)):
                    self.ax.text(self.parameters.left, self.temImage.sw_section_list[SectionText][0],
                                 'Layer ' + str(SectionText + 1) + ': ' +
                                 str('%.2f' % self.temImage.sw_thick_list[SectionText]) + ' nm', color='yellow',
                                 fontsize=7)
                    self.ax.text(self.parameters.left, self.temImage.sw_section_list[SectionText][0] + 40,
                                 'R: ' +
                                 str('%.2f' % self.temImage.sw_rough_list[SectionText]) + ' nm', color='yellow',
                                 fontsize=7)
            self.ax.text(self.parameters.right + 5,
                         (self.parameters.top + self.parameters.bottom) * 0.5,
                         'Thickness:' + str('%.3f' % self.temImage.sidewall) + ' nm', color='blue', fontsize=7)
            self.ax.text(self.parameters.right + 5, (self.parameters.top + self.parameters.bottom) * 0.5 + 40,
                         'Roughness: ' + str('%.3f' % self.temImage.roughness) + ' nm', color='blue', fontsize=7)

        plt.show()

    def grid_reset(self):
        self.parameters.field_position = int(self.field_ui.get())
        self.parameters.discontinuity = int(self.discontinuity_ui.get())
        self.parameters.noise_level = int(self.noise_ui.get())
# 以下函数，基本都是创建并响应UI里面的不同button的， 包括griding image, plot threshold, calculate thickness/void/sidewall.
# 如果已经有UI的话，不是很重要。
    def grid_plot(self):
        self.grid_reset()
        self.parameters.grid_parameter = int(self.grid_parameter_ui.get())
        self.temImage.get_grid(self.parameters)
        self.plot(grid=True)
        return

    def plot_threshold(self):
        if not self.threshold_ui:
            print("Invalid Threshold Entry")
            return

        threshold = int(self.threshold_ui.get())
        if not threshold or threshold < 0:
            print("Invalid threshold value")
            return

        self.parameters.set_threshold_value(threshold)
        self.parameters.set_threshold_list(self.temImage.height, self.temImage.width, self.temImage.data)
        if not self.var3.get():
            self.grid_parameter_ui.delete(0, 'end')
            self.grid_parameter_ui.insert(INSERT, self.threshold_ui.get())

        self.plot(threshold_flag=True, boundary_flag=True)
        return

    def plot_gray_threshold(self):
        if not self.gray_threshold_ui:
            print("Invalid Threshold Entry")
            return

        gray_threshold = int(self.gray_threshold_ui.get())

        if not gray_threshold or gray_threshold < 0:
            print("Invalid threshold value")
            return

        self.parameters.set_gray_threshold_value(gray_threshold)
        self.parameters.set_gray_threshold_list(self.temImage.height, self.temImage.width, self.temImage.data)

        self.plot(gray_flag=True)
        return

    def plot_bi_image(self):
        image = Image.new('RGB', (self.temImage.width, self.temImage.height), (255, 255, 255))

        draw = ImageDraw.Draw(image)
        self.parameters.set_pixels(self.temImage.height, self.temImage.width, self.temImage.data)
        for i in range(len(self.parameters.black_x)):
            draw.point((self.parameters.black_x[i], self.parameters.black_y[i]), fill=(64, 64, 64))
        for i in range(len(self.parameters.gray_x)):
            draw.point((self.parameters.gray_x[i], self.parameters.gray_y[i]), fill=(0, 102, 204))

        image.show()

    def get_boundary(self):
        self.refresh(t=False, b=False)

        left = self.boundary_ui[LEFT].get()
        right = self.boundary_ui[RIGHT].get()
        top = self.boundary_ui[TOP].get()
        bottom = self.boundary_ui[BOTTOM].get()
        angle = self.boundary_angle_ui.get()
        self.parameters.set_boundary_values(left, right, top, bottom)
        self.parameters.set_angle_value(angle)
        self.plot(boundary_flag=True)

        return

    def state_check(self):
        self.grid_reset()
        if self.parameters.threshold != int(self.threshold_ui.get()):
            self.parameters.set_threshold_value(int(self.threshold_ui.get()))
            self.parameters.set_threshold_list(self.temImage.height, self.temImage.width, self.temImage.data)
            if not self.var3.get():
                self.grid_parameter_ui.delete(0, 'end')
                self.grid_parameter_ui.insert(INSERT, self.threshold_ui.get())
                self.parameters.grid_parameter = int(self.grid_parameter_ui.get())
                self.temImage.get_grid(self.parameters)

        if not self.var3.get():
            self.parameters.grid_parameter = int(self.grid_parameter_ui.get())
            self.temImage.get_grid(self.parameters)
            if self.grid_parameter_ui.get() != self.threshold_ui.get():
                self.grid_parameter_ui.delete(0, 'end')
                self.grid_parameter_ui.insert(INSERT, self.threshold_ui.get())
                self.parameters.grid_parameter = int(self.grid_parameter_ui.get())
                self.temImage.get_grid(self.parameters)

        if self.var3.get():
            self.parameters.grid_parameter = int(self.grid_parameter_ui.get())
            self.temImage.get_grid(self.parameters)

    def get_thickness(self):
        self.state_check()
        self.refresh(t=False, b=False)
        with_thickness = 0
        if self.parameters.is_ready():
            self.temImage.calculate_thickness(self.parameters)
            self.thickness_output_ui()
            with_thickness = 1
        self.plot(value=with_thickness, threshold_flag=True, boundary_flag=True)
        return

    def get_void(self):
        self.state_check()
        self.refresh(t=False, b=False)
        with_void = 0
        if self.parameters.is_ready():
            self.temImage.calculate_thickness(self.parameters, voids=True)
            self.void_output_ui()
            with_void = 2
        self.plot(value=with_void, threshold_flag=True, boundary_flag=True)
        return

    def thickness_output_ui(self):
        pad_x, pad_y, inter_x = 5, 2, 5
        self.mean_thick_ui.delete(0, 'end')
        self.sd_thick_ui.delete(0, 'end')
        self.mean_thick_ui.insert(INSERT, str('%.3f' % self.temImage.thickness))
        self.sd_thick_ui.insert(INSERT, str('%.3f' % self.temImage.standard_deviation))
        self.table_ui('Thickness')
        Button(self.btm_frame, text='Column Distribution', command=lambda: self.distribution('Column')). \
            grid(sticky='EW', row=0, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)
        Button(self.btm_frame, text='Row Distribution', command=lambda: self.distribution('Row')). \
            grid(sticky='EW', row=1, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)

    def void_output_ui(self):
        pad_x, pad_y, inter_x = 5, 2, 5
        self.mean_void_ui.delete(0, 'end')
        self.sd_void_ui.delete(0, 'end')
        self.mean_void_ui.insert(INSERT, str('%.3f' % (100 * self.temImage.void_percentage)))
        self.sd_void_ui.insert(INSERT, str('%.3f' % self.temImage.void_sd))
        self.table_ui('Void%')
        Button(self.btm_frame, text='Column Distribution', command=lambda: self.distribution('Column', void=True)). \
            grid(sticky='EW', row=0, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)
        Button(self.btm_frame, text='Row Distribution', command=lambda: self.distribution('Row', void=True)). \
            grid(sticky='EW', row=1, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)

    def get_sidewall(self):
        self.state_check()
        self.refresh(t=False, b=False)
        with_side = 0
        if self.parameters.is_ready():
            with_side = 3
            self.temImage.calculate_sidewall(self.parameters, self.var2.get())
            self.sidewall_ui()
        self.plot(value=with_side, threshold_flag=True, boundary_flag=True)
        return

    def sidewall_ui(self):
        pad_x, pad_y, inter_x = 5, 2, 5
        self.sw_thickness_ui.delete(0, 'end')
        self.sw_roughness_ui.delete(0, 'end')
        self.sw_thickness_ui.insert(INSERT, str('%.3f' % self.temImage.sidewall))
        self.sw_roughness_ui.insert(INSERT, str('%.3f' % self.temImage.roughness))

        if self.var2.get() == 0:
            self.temImage.calculate_sidewall_section(self.parameters)
            self.setup_sw_frame()
            Button(self.btm_frame, text='Thickness', command=lambda: self.distribution('Row', sidewall=1)). \
                grid(sticky='EW', row=0, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)
            Button(self.btm_frame, text='Roughness', command=lambda: self.distribution('Row', sidewall=2)). \
                grid(sticky='EW', row=1, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)

    def setup_sw_frame(self):
        pad_x, pad_y, inter_x = 5, 2, 5
        self.sw_frame.grid(row=3, column=4, sticky='EW', padx=5, pady=2)
        height = len(self.temImage.sw_thick_list) + 3
        width = 3
        b = [[0 for x in range(width)] for y in range(height)]
        b[0][0] = 'Sidewall'
        for i in range(1, height - 2):
            b[i][0] = 'Row %s' % str(i)
            b[i][1] = '%.3f' % self.temImage.sw_thick_list[i - 1]
            b[i][2] = '%.3f' % self.temImage.sw_rough_list[i - 1]
        b[height - 1][0] = 'SD'
        b[height - 2][0] = 'Average'
        if len(self.temImage.sw_thick_list) > 1:
            b[height - 1][1] = '%.3f' % statistics.stdev(self.temImage.sw_thick_list)
            b[height - 2][1] = '%.3f' % statistics.mean(self.temImage.sw_thick_list)
            b[height - 1][2] = '%.3f' % statistics.stdev(self.temImage.sw_rough_list)
            b[height - 2][2] = '%.3f' % statistics.mean(self.temImage.sw_rough_list)
        else:
            b[height - 1][1] = 'NA'
            b[height - 2][1] = 'NA'
            b[height - 1][2] = 'NA'
            b[height - 2][2] = 'NA'
        b[0][1] = 'Thickness'
        b[0][2] = 'Roughness'

        self.sw_table = [[0 for x in range(width)] for y in range(height)]

        for i in range(height):
            for j in range(width):
                self.sw_table[i][j] = Entry(self.sw_frame)
                self.sw_table[i][j].place(x=j * 70, y=i * 20, width=70)
                self.sw_table[i][j].insert(INSERT, b[i][j])

        Button(self.btm_frame, text='Output', command=lambda: self.out2csv(b, 'Sidewall')). \
            grid(sticky='EW', row=2, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)

    def table_ui(self, s):
        pad_x, pad_y, inter_x = 5, 2, 5
        column = []
        row = []
        matrix = []
        average = 0
        if s == 'Thickness':
            column = self.temImage.thickness_column
            row = self.temImage.thickness_list
            matrix = self.temImage.thickness_matrix
            average = self.temImage.thickness
        if s == 'Void%':
            column = self.temImage.void_column
            row = self.temImage.void_list
            matrix = self.temImage.void_matrix
            average = self.temImage.void_percentage

        height = len(self.temImage.section_list) + 2
        width = 12
        b = [[0 for x in range(width)] for y in range(height)]
        self.output_table = [[0 for x in range(width)] for y in range(height)]

        b[0][0] = s
        for j in range(1, width - 1):
            b[0][j] = 'Col %s' % str(j)
            b[height - 1][j] = '%.3f' % column[j - 1]
        for i in range(1, height - 1):
            b[i][0] = 'Row %s' % str(i)
            b[i][width - 1] = '%.3f' % row[i - 1]
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                b[i][j] = '%.3f' % matrix[i - 1][j - 1]
        b[0][width - 1] = 'C-Average'
        b[height - 1][0] = 'R-Average'
        b[height - 1][width - 1] = '%.3f' % average
        self.table_frame.grid(row=3, columnspan=3, sticky='EW', padx=5, pady=2)
        for i in range(height):
            for j in range(width):
                self.output_table[i][j] = Entry(self.table_frame)
                if j == 0:
                    self.output_table[i][j].place(x=j * 70, y=i * 20, width=70)
                elif j == width - 1:
                    self.output_table[i][j].place(x=120 + (j - 2) * 50, y=i * 20, width=70)
                else:
                    self.output_table[i][j].place(x=70 + (j - 1) * 50, y=i * 20, width=50)
                self.output_table[i][j].insert(INSERT, b[i][j])

        Button(self.btm_frame, text='Output', command=lambda: self.out2csv(b, s)). \
            grid(sticky='EW', row=2, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)

    def out2csv(self, data, name):
        my_data = data
        label, summary = [], []
        if name == 'Thickness':
            label = [[self.temImage.filename], ['Thickness Analysis']]
            summary = [['Thickness', 'SD'], [self.temImage.thickness, self.temImage.standard_deviation]]
        if name == 'Void%':
            label = [[self.temImage.filename], ['Void Analysis']]
            summary = [['Void Percentage', 'SD'], [self.temImage.void_percentage, self.temImage.void_sd]]
        if name == 'Sidewall':
            label = [[self.temImage.filename], ['Sidewall Analysis']]
            summary = [['Thickness', 'Roughness'], [self.temImage.sidewall, self.temImage.roughness]]

        my_file = open('Output.csv', 'a')
        with my_file:
            writer = csv.writer(my_file)
            writer.writerows(label)
            writer.writerows(summary)
            writer.writerows(my_data)

    def distribution(self, name, void=False, sidewall=0):
        y = []
        if name == 'Column':
            y = self.temImage.thickness_column
            if void:
                y = [100 * float(n) for n in self.temImage.void_column]
        if name == 'Row':
            y = self.temImage.thickness_list
            if void:
                y = [100 * float(n) for n in self.temImage.void_list]
            if sidewall == 1:
                y = self.temImage.sw_thick_list
            if sidewall == 2:
                y = self.temImage.sw_rough_list

        length = len(y)
        width = 0.4
        x = [n for n in range(length)]
        plt.figure('%s Distribution' % name)
        plt.title('%s Distribution' % name)
        plt.ylabel('Thickness (nm)')
        if void:
            plt.ylabel('Void Percentage (%)')
        if sidewall == 1:
            plt.ylabel('Thickness (nm)')
        if sidewall == 2:
            plt.ylabel('Roughness (nm)')
        plt.xlabel(name)
        plt.bar(x, y, width)
        x_ticks = [str(n + 1) for n in range(length)]
        plt.xticks(x, x_ticks)
        y_min, y_max = plt.ylim()
        y_min = min(y) / 1.5
        plt.ylim(y_min, y_max)
        plt.show()

    def mouse_wheel(self, event):
        global connect_flag
        if connect_flag:
            count = int(self.threshold_ui.get())
            # respond to Linux or Windows wheel event
            if event.button == 'up':
                # deal with zoom in
                count -= 1
            elif event.button == 'down':
                # deal with zoom out
                count += 1

            self.threshold_ui.delete(0, 'end')
            self.threshold_ui.insert(INSERT, count)
            self.threshold_scatter(region=False)

        else:
            return

    def setup_left1_frame(self):
        pad_x = 5
        pad_y = 2
        inter_x = 5
        Label(self.left_frame1, text="Threshold").grid(sticky='W', row=0, column=0, padx=pad_x, pady=pad_y)
        self.threshold_ui = Entry(self.left_frame1)
        self.threshold_ui.insert(INSERT, 50)
        self.threshold_ui.grid(sticky='W', row=0, column=1, padx=pad_x, pady=pad_y)
        Button(self.left_frame1, text='Plot Threshold', command=lambda: self.plot_threshold()). \
            grid(sticky='EW', row=0, column=2, padx=pad_x, pady=pad_y, ipadx=inter_x)

        Label(self.left_frame1, text="Scale Bar (nm)").grid(sticky='W', row=1, column=0, padx=pad_x, pady=pad_y)
        self.scale_bar_ui = Entry(self.left_frame1)
        self.scale_bar_ui.grid(sticky='W', row=1, column=1, padx=pad_x, pady=pad_y)
        cvt = 'Convert ratio: '
        Label(self.left_frame1, text=cvt).grid(sticky='W', row=2, column=0, padx=pad_x, pady=pad_y)

        Button(self.left_frame1, text='Update', command=lambda: self.update_ratio_ui()). \
            grid(sticky='EW', row=1, column=2, padx=pad_x, pady=pad_y, ipadx=inter_x)

        Checkbutton(self.left_frame1, text="lock", variable=self.var1). \
            grid(sticky='NSEW', row=1, column=3, padx=pad_x, pady=pad_y)

        Checkbutton(self.left_frame1, text="draw", variable=self.var4). \
            grid(sticky='NSEW', row=0, column=3, padx=pad_x, pady=pad_y)

        Button(self.left_frame1, text='Clear', command=lambda: self.clear_draw()). \
            grid(sticky='EW', row=0, column=4, padx=pad_x, pady=pad_y, ipadx=inter_x)

    def setup_left2_frame(self):
        pad_x = 5
        pad_y = 2
        inter_x = 5
        Label(self.left_frame2, text="Field Region").grid(sticky='W', row=0, column=0, padx=pad_x - 2, pady=pad_y)
        self.field_ui = Entry(self.left_frame2)
        self.field_ui.grid(sticky='W', row=0, column=1, padx=pad_x, pady=pad_y)
        self.field_ui.insert(INSERT, 0)

        Label(self.left_frame2, text="Discontinuity").grid(sticky='W', row=0, column=2, padx=pad_x - 2, pady=pad_y)
        self.discontinuity_ui = Entry(self.left_frame2)
        self.discontinuity_ui.grid(sticky='W', row=0, column=3, padx=pad_x, pady=pad_y)
        self.discontinuity_ui.insert(INSERT, 0)

        Label(self.left_frame2, text="Noise Level").grid(sticky='W', row=1, column=0, padx=pad_x - 2, pady=pad_y)
        self.noise_ui = Entry(self.left_frame2)
        self.noise_ui.grid(sticky='W', row=1, column=1, padx=pad_x, pady=pad_y)
        self.noise_ui.insert(INSERT, 2)

        #Label(self.left_frame2, text="Gray Threshold").grid(sticky='W', row=0, column=0, padx=pad_x - 2, pady=pad_y)
        #self.gray_threshold_ui = Entry(self.left_frame2)
        #self.gray_threshold_ui.insert(INSERT, 125)
        #self.gray_threshold_ui.grid(sticky='W', row=0, column=1, padx=pad_x, pady=pad_y)
        #Button(self.left_frame2, text='Plot Threshold', command=lambda: self.plot_gray_threshold()). \
        #    grid(sticky='EW', row=0, column=2, padx=pad_x, pady=pad_y, ipadx=inter_x)

        #Button(self.left_frame2, text='Bi-Image', command=lambda: self.plot_bi_image()). \
        #    grid(sticky='EW', row=0, column=3, padx=pad_x, pady=pad_y, ipadx=inter_x + 7)
        Button(self.left_frame2, text='Image Grid', command=lambda: self.grid_plot()). \
            grid(sticky='EW', row=2, column=0, padx=pad_x, pady=pad_y, ipadx=inter_x - 1)

        self.grid_parameter_ui.grid(sticky='W', row=2, column=1, padx=pad_x, pady=pad_y)
        self.grid_parameter_ui.delete(0, 'end')
        self.grid_parameter_ui.insert(INSERT, 50)
        Checkbutton(self.left_frame2, text="lock", variable=self.var3). \
            grid(sticky='NSEW', row=2, column=2, padx=pad_x, pady=pad_y)

    def setup_right_frame(self):
        labels = [LEFT, RIGHT, TOP, BOTTOM]
        pad_x = 5
        pad_y = 5
        inter_x = 5
        i = 0
        for label_item in labels:
            Label(self.right_frame, text=label_item).grid(sticky='W', row=i, column=0, padx=pad_x)
            display_entry = Entry(self.right_frame)
            display_entry.grid(sticky='W', row=i, column=1, padx=pad_x)
            self.boundary_ui[label_item] = display_entry
            i += 1

        Label(self.right_frame, text='Angle').grid(sticky='W', row=4, column=0, padx=pad_x)

        self.boundary_angle_ui.insert(INSERT, 0)
        self.boundary_angle_ui.grid(sticky='W', row=4, column=1, padx=pad_x)

        Button(self.right_frame, text='Get Boundary', command=lambda: self.get_boundary()) \
            .grid(sticky='W', row=5, padx=pad_x, pady=pad_y, ipadx=inter_x)

    def setup_btm_frame(self):
        pad_x = 5
        pad_y = 2
        inter_x = 5
        Button(self.btm_frame, text='Thickness', command=lambda: self.get_thickness()). \
            grid(sticky='EW', row=0, padx=pad_x, pady=pad_y, ipadx=inter_x)
        Button(self.btm_frame, text='Void Percentage', command=lambda: self.get_void()). \
            grid(sticky='EW', row=1, padx=pad_x, pady=pad_y, ipadx=inter_x)
        Button(self.btm_frame, text='Sidewall/Field', command=lambda: self.get_sidewall()). \
            grid(sticky='EW', row=2, padx=pad_x, pady=pad_y, ipadx=inter_x)

        Label(self.btm_frame, text='Thickness (nm):').grid(sticky='W', row=0, column=1, padx=pad_x)
        self.mean_thick_ui.grid(sticky='W', row=0, column=2, padx=pad_x)
        Label(self.btm_frame, text='Standard Deviation:').grid(sticky='W', row=0, column=3, padx=pad_x)
        self.sd_thick_ui.grid(sticky='W', row=0, column=4, padx=pad_x)

        Label(self.btm_frame, text='Void Percentage:').grid(sticky='W', row=1, column=1, padx=pad_x)
        self.mean_void_ui.grid(sticky='W', row=1, column=2, padx=pad_x)
        Label(self.btm_frame, text='Standard Deviation:').grid(sticky='W', row=1, column=3, padx=pad_x)
        self.sd_void_ui.grid(sticky='W', row=1, column=4, padx=pad_x)

        Label(self.btm_frame, text='Thickness (nm):').grid(sticky='W', row=2, column=1, padx=pad_x)
        self.sw_thickness_ui.grid(sticky='W', row=2, column=2, padx=pad_x)
        Label(self.btm_frame, text='Roughness (nm):').grid(sticky='W', row=2, column=3, padx=pad_x)
        self.sw_roughness_ui.grid(sticky='W', row=2, column=4, padx=pad_x)

        Button(self.btm_frame, text='Column Distribution'). \
            grid(sticky='EW', row=0, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)
        Button(self.btm_frame, text='Row Distribution'). \
            grid(sticky='EW', row=1, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)
        Button(self.btm_frame, text='Output'). \
            grid(sticky='EW', row=2, column=5, padx=pad_x, pady=pad_y, ipadx=inter_x)

        Checkbutton(self.btm_frame, text="Field Region", variable=self.var2). \
            grid(sticky='NSEW', row=3, column=0, padx=pad_x, pady=pad_y)

    def construct_ui(self):
        self.root.geometry('840x450+50+50')
        self.root.title("Image Analysis")

        menu_bar = Menu(self.root)
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file_ui)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu_bar)

        self.left_frame1.grid(row=0, column=0, sticky='EW', padx=5, pady=2)
        self.left_frame2.grid(row=1, column=0, sticky='EW', padx=5, pady=2)
        self.right_frame.grid(row=0, column=1, rowspan=2, sticky='NSEW', padx=5, pady=2)
        self.btm_frame.grid(row=2, columnspan=2, sticky='EW', padx=5, pady=2)

        self.setup_left1_frame()
        self.setup_left2_frame()
        self.setup_right_frame()
        self.setup_btm_frame()

        self.root.mainloop()


connect_flag = False
my_ui = UI()
my_ui.construct_ui()
