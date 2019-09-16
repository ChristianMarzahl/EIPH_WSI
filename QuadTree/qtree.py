import numpy as np
from matplotlib import pyplot as plt
from random import randint
import matplotlib.patches as mpatch
from matplotlib import colors as mcolors



class Leaf:

    def __init__(self, x:int, y:int, label:str, priority:float=1):
        self._x = x
        self._y = y
        self._label = label
        self._priority = priority

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def label(self):
        return self._label

    @property
    def priority(self):
        return self._priority

    @priority.setter
    def priority(self, value):
        self._priority = value

    def __str__(self):
        return 'X: {} Y: {} Label: {}'.format(self._x, self._y, self._label, self._priority)


class QuadTree:

    def __init__(self, data: list, mins: tuple, maxs: tuple, depth: int
                 , divide_by: callable=None, metric_func: callable=None):
        self.data = data
        self.depth = depth

        self.metric_func = metric_func if callable(metric_func) else self._probability

        if mins is None:
            mins = (min(self.data, key=lambda x: x.x).x, min(self.data, key=lambda x: x.y).y)
        if maxs is None:
            maxs = (max(self.data, key=lambda x: x.x).x, max(self.data, key=lambda x: x.y).y)

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.sizes = self.maxs - self.mins

        self.children = []

        mids = 0.5 * (self.mins + self.maxs)
        xmin, ymin = self.mins
        xmax, ymax = self.maxs
        xmid, ymid = mids

        divide = divide_by(self) if callable(divide_by) else depth > 0
        if divide:
            data_q1 = [leaf for leaf in data if (leaf.x < mids[0]) & (leaf.y < mids[1])]
            data_q2 = [leaf for leaf in data if (leaf.x < mids[0]) & (leaf.y >= mids[1])]
            data_q3 = [leaf for leaf in data if (leaf.x >= mids[0]) & (leaf.y < mids[1])]
            data_q4 = [leaf for leaf in data if (leaf.x >= mids[0]) & (leaf.y >= mids[1])]

            if len(data_q1) > 0:
                self.children.append(QuadTree(data_q1,
                                              [xmin, ymin], [xmid, ymid],
                                              depth - 1, divide_by=divide_by,
                                              metric_func=self.metric_func))
            if len(data_q2) > 0:
                self.children.append(QuadTree(data_q2,
                                              [xmin, ymid], [xmid, ymax],
                                              depth - 1, divide_by=divide_by,
                                              metric_func=self.metric_func))
            if len(data_q3) > 0:
                self.children.append(QuadTree(data_q3,
                                              [xmid, ymin], [xmax, ymid],
                                              depth - 1, divide_by=divide_by,
                                              metric_func=self.metric_func))
            if len(data_q4) > 0:
                self.children.append(QuadTree(data_q4,
                                              [xmid, ymid], [xmax, ymax],
                                              depth - 1, divide_by=divide_by,
                                              metric_func=self.metric_func))

    @property
    def metric(self):
        return self.metric_func(self)

    def _probability(self, o):
        priorities = [np.mean([d.priority for d in c.data]) for c in self.children]
        return [p/np.sum(priorities) for p in priorities]

    def draw_rectangle(self, ax, depth:int, prob:float=1, edgecolor='#000000', use_percentages=False):
        """Recursively plot a visualization of the quad tree region"""
        if (depth is None or depth > 0) and len(self.children) > 0:
            color = 'black'# np.random.choice(list(mcolors.CSS4_COLORS), 1)[0]
            for child, prob in zip(self.children, self.metric):
                if use_percentages: prob = prob / sum(self.metric) * 100
                child.draw_rectangle(ax, depth - 1, prob, color, use_percentages)
        else:
            rect = mpatch.Rectangle(self.mins, *self.sizes, zorder=2, ec=edgecolor, fc='none')
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            ax.add_artist(rect)

            if use_percentages:
                ax.annotate('{:1.1f}%'.format(prob), (cx, cy), color='red', weight='bold',
                                 ha='center', va='center') #fontsize=12,
            else:
                ax.annotate('{:1.2f}'.format(prob), (cx, cy), color='black', weight='bold',
                           ha='center', va='center') #fontsize=12,



    def __str__(self):
        return 'Min: {} Max: {} Size: {} Childs: {} '.format(self.mins, self.maxs, self.sizes, len(self.data))