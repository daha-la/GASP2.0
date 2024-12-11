#!/usr/bin/env python3
# coding=utf-8
import sys
import curses
import atexit
import time
from pynput import keyboard, mouse
from degnutil import math_util as ma, input_output as io

"""
This script is for curses utilities so things to with drawing a screen with paging functionality and so on.
"""


def window():
    """
    Create a new screen for printing.
    This also registers to end screen when closing application.
    :return: window instance
    """
    win = curses.initscr()
    curses.noecho()
    curses.cbreak()
    win.keypad(True)
    win.erase()
    io.remove_stdout()
    
    @atexit.register
    def end(): end_window(win)
    
    return win


def end_window(win):
    """Remove screen instance and reset terminal to how it was before. """
    win.keypad(False)
    curses.initscr()
    curses.nocbreak()
    curses.flushinp()
    curses.endwin()
    curses.echo()
    sys.stdout.flush()


def get_width():
    return curses.COLS


def get_height():
    return curses.LINES


def updown(key):
    if key == keyboard.Key.up: return -1
    elif key == keyboard.Key.down: return 1
    return 0


def leftright(key):
    if key == keyboard.Key.left: return -1
    elif key == keyboard.Key.right: return 1
    return 0


def add_string(win, string, x=0, y=0, screen_x=0, screen_y=0, attr=None):
    """
    Add a string to be drawn to a screen. 
    Gets limited horizontally to fit in screen.
    :param win: screen to draw to.
    :param string: string to draw.
    :param x: character position to draw at.
    :param y: line to draw on.
    :param screen_x: 
    :param screen_y: 
    :param attr: attributes for formatting e.g. curses.A_BOLD for bold text
    :return: 
    """
    # we paint relative to screen position
    x -= screen_x
    y -= screen_y
    w = get_width()
    # limit end
    if x + len(string) > w: string = string[:w-x]
    # limit start
    if x < 0:
        string = string[-x:]
        x = 0
    
    try:
        if attr is None: win.addstr(y, x, string)
        else: win.addstr(y, x, string, attr)
    # curses raises except when drawing to lower right corner but only after drawing to it
    except curses.error: pass


def add_lines(win, lines, screen_x=0, screen_y=0):
    """
    Add lines to be drawn. Run screen.erase() before and screen.refresh() after.
    :param win: screen object the lines are added to.
    :param lines: list of strings to add in the order given. Clipped at the length of the screen.
    :param screen_x: starting column to draw aka. what is the x of the screen view into the file.
    :param screen_y: starting row to draw aka. what is the y of the screen view into the file.
    :return: None
    """
    for i, line in enumerate(lines[screen_y:screen_y + get_height()]):
        add_string(win, line, y=i, screen_x=screen_x)


def _str(val, **kwargs):
    """
    Just a version of str that allows and ignores keyword args.
    The point is to use it as a formatter function for various inputs
    """
    return str(val)


def add_array(win, array, cell_widths, screen_x=0, screen_y=0, formatter=_str, space=1):
    """
    Add a subsection of a table to screen drawing.
    :param win: screen object to add table section to.
    :param array: numpy array to add a subsection of or the whole thing if there's room.
    :param cell_widths: list of int widths for each column.
    :param screen_x: int x of top left character to draw aka. position of screen view into file
    :param screen_y: int y of top left character to draw aka. position of screen view into file
    :param formatter: function to use on a cell before it is drawn.
    :param space: int value for min number of characters of space left between columns.
    :return: None
    """
    x = 0
    for c in range(0, array.shape[1]):
        if x > screen_x + get_width(): return
        
        w = cell_widths[c]
        # only print cell (or part of cell) if its visible
        if x + w > screen_x:
            # draw a column
            for r in range(screen_y, min(array.shape[0], screen_y + get_height())):
                add_string(win, formatter(array[r, c])[:w], x, r, screen_x, screen_y)
        
        # move x to next column start
        x += w + space

 
def add_table(win, table, cell_widths, screen_x=0, screen_y=0, formatter=_str, space=1):
    """
    Add a subsection of a table to screen drawing.
    :param win: screen object to add table section to.
    :param table: pandas dataframe to add a subsection of or the whole thing if there's room.
    :param cell_widths: int cell width, or list of widths for each column.
    :param screen_x: the row index of the top left character visible.
    :param screen_y: the column index of the top left character visible.
    :param formatter: function to use on a cell before it is drawn.
    :param space: int value for min number of characters of space left between columns.
    :return: None
    """
    index_w = cell_widths[0]
    x = index_w + space
    for c in range(0, table.shape[1]):
        if x > screen_x + get_width(): break

        # make room for the index width
        w = cell_widths[c + 1]
        # only print column (or part of column) if its visible
        if x + w > screen_x:

            # draw a column name
            add_string(win, formatter(table.columns[c])[:w], x=x, screen_x=screen_x, attr=curses.A_BOLD)
            
            # draw a column
            for r in range(screen_y, min(table.shape[0], screen_y + get_height() - 1)):
                add_string(win, formatter(table.iloc[r, c])[:w], x, r + 1, screen_x, screen_y)

        # move x to next column start
        x += w + space

    # make sure index column is clear
    for r in range(0, get_height()):
        add_string(win, " " * index_w, y=r)

    # draw table index
    # row has to be set to allow for extra space above index
    for r in range(screen_y, min(table.shape[0], screen_y + get_height() - 1)):
        add_string(win, formatter(table.index[r])[:index_w], y=r + 1, screen_y=screen_y, attr=curses.A_BOLD)


def is_pandas(a):
    return hasattr(a, "index")



class View:
    """
    A simple screen object to display a text file.
    Can be extended for more specialized pagers.
    """
    
    delta_time = 1./60.
    
    def __init__(self, content):
        self.win = window()
        self.dirty = True
        self.content = content
        # position of upper left corner of screen view into file
        self.x, self.y, self.width, self.height = 0, 0, self._get_width(), self._get_height()
        self.keyboard = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.mouse = mouse.Listener(on_scroll=self.on_scroll)
    

    def __del__(self):
        end_window(self.win)

    
    def draw(self):
        """Add text to screen object for drawing. """
        add_lines(self.win, self.content, self.x, self.y)

    
    def on_scroll(self, x, y, dx, dy):
        self.translate(-dx, -dy)
    

    def on_press(self, key):
        """
        Method for handling key input. Return False to terminate.
        :param key: int key pressed, function fires repeatedly when pressed.
        :return: False to terminate
        """
        self.translate(leftright(key), updown(key))


    def on_release(self, key):
        """
        Method for handling key input. Return False to terminate.
        :param key: int key released, function fires once when released.
        :return: False to terminate
        """
        try: char = key.char
        except AttributeError: pass
        else:
            char = char.lower()
            if char == 'q':
                return False


    def render(self):
        """A loop to keep redrawing until user gives input to exit. """
        self.keyboard.start()
        self.mouse.start()
        while self.keyboard.is_alive():
            if self.dirty:
                # important: set dirty false before draw call 
                # so in case we update x and y while drawing it will still be dirty
                self.dirty = False
                self.win.erase()
                self.draw()
                self.win.refresh()
            time.sleep(self.delta_time)
    
    
    def clip_x(self, x):
        # we cannot move x away from 0 if the content does not extent beyond the visible area
        if self.width < get_width(): return 0
        return ma.clip(x, 0, self.width - get_width())
    
    
    def clip_y(self, y):
        if self.height < get_height(): return 0
        return ma.clip(y, 0, self.height - get_height())
    
    
    def translate(self, dx, dy):
        x = self.clip_x(self.x + dx)
        y = self.clip_y(self.y + dy)
        if self.x != x or self.y != y:
            self.x, self.y = x, y
            self.dirty = True
    
    
    # getters
    
    def _get_height(self):
        return len(self.content)
    
    
    def _get_width(self):
        """
        Calculate number of columns (characters) as the maximum line length
        :return: 
        """
        return max(len(l) for l in self.content)


class ViewTable(View):
    """
    A simple screen object to display a table file.
    Can be extended for more specialized pagers.
    """
    
    space = 1
    
    def __init__(self, content, cell_width):
        if isinstance(cell_width, list): self.cell_widths = cell_width
        else:
            self.cell_widths = [cell_width for _ in range(content.shape[1])]
            if is_pandas(content): self.cell_widths.append(cell_width)
            
        super().__init__(content)
        self.add_func = add_table if is_pandas(self.content) else add_array
    
    
    def draw(self):
        self.add_func(self.win, self.content, self.cell_widths, self.x, self.y, space=self.space)

    
    def _get_height(self):
        return self.content.shape[0]
    
    
    def _get_width(self):
        return sum(self.cell_widths) + self.space * (len(self.cell_widths) - 1)
    
    