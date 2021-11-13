# This file is part of GridCal.
#
# GridCal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GridCal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GridCal.  If not, see <http://www.gnu.org/licenses/>.

import math
from PySide2 import QtWidgets, QtCore, QtGui


'''
See
https://stackoverflow.com/questions/44246283/how-to-add-a-arrow-head-to-my-line-in-pyqt4
'''


class Path(QtWidgets.QGraphicsPathItem):

    def __init__(self, source: QtCore.QPointF = None, destination: QtCore.QPointF = None, *args, **kwargs):
        super(Path, self).__init__(*args, **kwargs)

        self.pos1 = source
        self.pos2 = destination

        self._arrow_height = 5
        self._arrow_width = 4

    def setBeginPos(self, point: QtCore.QPointF):
        self.pos1 = point

    def setEndPos(self, point: QtCore.QPointF):
        self.pos2 = point

    def direct_path(self):
        path = QtGui.QPainterPath(self.pos1)
        path.lineTo(self.pos2)
        return path

    def square_path(self):
        s = self.pos1
        d = self.pos2

        mid_x = s.x() + ((d.x() - s.x()) * 0.5)

        path = QtGui.QPainterPath(QtCore.QPointF(s.x(), s.y()))
        path.lineTo(mid_x, s.y())
        path.lineTo(mid_x, d.y())
        path.lineTo(d.x(), d.y())

        return path

    def bezier_path(self):
        s = self.pos1
        d = self.pos2

        source_x, source_y = s.x(), s.y()
        destination_x, destination_y = d.x(), d.y()

        dist = (d.x() - s.x()) * 0.5

        cpx_s = +dist
        cpx_d = -dist
        cpy_s = 0
        cpy_d = 0

        if (s.x() > d.x()) or (s.x() < d.x()):
            cpx_d *= -1
            cpx_s *= -1

            cpy_d = ((source_y - destination_y) / math.fabs((source_y - destination_y) if (source_y - destination_y) != 0 else 0.00001)) * 150

            cpy_s = ((destination_y - source_y) / math.fabs((destination_y - source_y) if (destination_y - source_y) != 0 else 0.00001)) * 150

        path = QtGui.QPainterPath(self.pos1)

        path.cubicTo(destination_x + cpx_d, destination_y + cpy_d, source_x + cpx_s, source_y + cpy_s,
                     destination_x, destination_y)

        return path

    def arrow_calc(self, start_point=None, end_point=None):  # calculates the point where the arrow should be drawn

        try:
            startPoint, endPoint = start_point, end_point

            if start_point is None:
                startPoint = self.pos1

            if endPoint is None:
                endPoint = self.pos2

            dx, dy = startPoint.x() - endPoint.x(), startPoint.y() - endPoint.y()

            leng = math.sqrt(dx ** 2 + dy ** 2)
            normX, normY = dx / leng, dy / leng  # normalize

            # perpendicular vector
            perpX = -normY
            perpY = normX

            leftX = endPoint.x() + self._arrow_height * normX + self._arrow_width * perpX
            leftY = endPoint.y() + self._arrow_height * normY + self._arrow_width * perpY

            rightX = endPoint.x() + self._arrow_height * normX - self._arrow_height * perpX
            rightY = endPoint.y() + self._arrow_height * normY - self._arrow_width * perpY

            point2 = QtCore.QPointF(leftX, leftY)
            point3 = QtCore.QPointF(rightX, rightY)

            return QtGui.QPolygonF([point2, endPoint, point3])

        except (ZeroDivisionError, Exception):
            return None

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:

        painter.setRenderHint(painter.Antialiasing)

        painter.pen().setWidth(2)
        painter.setBrush(QtCore.Qt.NoBrush)

        path = self.direct_path()
        # path = self.bezier_path()
        # path = self.square_path()
        painter.drawPath(path)
        self.setPath(path)

        # triangle_source = self.arrowCalc(path.pointAtPercent(0.1), self._sourcePoint)  # change path.PointAtPercent() value to move arrow on the line
        triangle_source = self.arrow_calc(path.pointAtPercent(0.5), path.pointAtPercent(0.51))
        if triangle_source is not None:
            painter.drawPolyline(triangle_source)
