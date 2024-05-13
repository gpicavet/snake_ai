import math
import random
from collections import namedtuple
from enum import Enum
from random import randrange
from typing import Optional, Tuple

random.seed(1)


class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


Point = namedtuple('Point', 'x, y')


class Snake:
    body: list[Point]
    dir: Point
    age: int
    starve_count: int

    def __init__(self, pos: Point, dir: Point):
        self.body = [pos]
        self.dir = dir
        self.age = 0
        self.starve_count = 100

    def move(self):
        self.age += 1
        self.starve_count -= 1
        self.body = [Point(self.body[0].x + self.dir.x, self.body[0].y + self.dir.y)] + self.body[:-1]

    def grow(self, tail: Point):
        self.starve_count = 0
        self.body.append(tail)


class Game:
    apple: Optional[Point]
    snake: Optional[Snake]

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.apple = None
        self.snake = None

    def start(self):
        sx = 2 + randrange(self.w - 4)
        sy = 2 + randrange(self.h - 4)
        d = Point(-1, 0) if sx > self.w / 2 else Point(1, 0)
        self.snake = Snake(Point(sx, sy), d)

        self.new_apple()

    def new_apple(self):
        possible = [Point(x, y) for x in range(self.w) for y in range(self.h) if Point(x, y) not in self.snake.body]
        self.apple = possible[randrange(len(possible))]

    def step(self) -> Tuple[int, bool, int]:
        tail = self.snake.body[-1]

        self.snake.move()
        if self.is_collision() or self.snake.starve_count == 0:
            return -10, True, self.score()

        head = self.snake.body[0]
        if head == self.apple:
            self.new_apple()
            self.snake.grow(tail)
            return +10, False, self.score()

        return 0, False, self.score()

    def is_collision(self, pt: Optional[Point] = None):
        if not pt:
            pt = self.snake.body[0]
        if self.is_wall(pt):
            return True
        if self.is_body(pt):
            return True
        return False

    def is_wall(self, pt: Point):
        return pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h

    def is_body(self, pt: Point):
        return pt in self.snake.body[1:]

    def score(self):
        return len(self.snake.body)

    def dir(self, dir: Direction):
        [x, y] = self.snake.dir
        if dir == Direction.LEFT:
            self.snake.dir = Point(y, -x)
        if dir == Direction.RIGHT:
            self.snake.dir = Point(-y, x)

    def dir_human(self, dir: Direction):
        if dir == Direction.UP:
            self.snake.dir = Point(0, -1)
        if dir == Direction.DOWN:
            self.snake.dir = Point(0, +1)
        if dir == Direction.LEFT:
            self.snake.dir = Point(-1, 0)
        if dir == Direction.RIGHT:
            self.snake.dir = Point(+1, 0)

    def get_state(self):
        head = self.snake.body[0]

        dir = self.snake.dir

        def wall_dist(d: Point) -> float:
            dst = 0
            if d.x > 0:
                dst = self.w - head.x
            elif d.x < 0:
                dst = head.x + 1
            elif d.y > 0:
                dst = self.h - head.y
            elif d.y < 0:
                dst = head.y + 1

            return 1 / (1 + dst)

        def body_dist(d: Point) -> float:
            dst = self.w * self.w + self.h * self.h

            #left
            if d.x < 0 and d.y == 0:
                for b in self.snake.body[1:]:
                    if b.y == head.y and b.x < head.x:
                        dst = min(dst, (head.x - b.x) ** 2)
            #up-left
            if d.x < 0 and d.y < 0:
                for b in self.snake.body[1:]:
                    if b.y - b.x == head.y - head.x and b.y < head.y:
                        dst = min(dst, (head.y - b.y) ** 2 + (head.x - b.x) ** 2)
            #up
            if d.x == 0 and d.y < 0:
                for b in self.snake.body[1:]:
                    if b.x == head.x and b.y < head.y:
                        dst = min(dst, (head.y - b.y) ** 2)
            #up-right
            if d.x > 0 and d.y < 0:
                for b in self.snake.body[1:]:
                    if b.y + b.x == head.y + head.x and b.y < head.y:
                        dst = min(dst, (head.y - b.y) ** 2 + (head.x - b.x) ** 2)
            #right
            if d.x > 0 and d.y == 0:
                for b in self.snake.body[1:]:
                    if b.y == head.y and b.x > head.x:
                        dst = min(dst, (head.x - b.x) ** 2)
            #bottom-right
            if d.x > 0 and d.y > 0:
                for b in self.snake.body[1:]:
                    if b.y - b.x == head.y - head.x and b.y > head.y:
                        dst = min(dst, (head.y - b.y) ** 2 + (head.x - b.x) ** 2)

            #bottom
            if d.x == 0 and d.y > 0:
                for b in self.snake.body[1:]:
                    if b.x == head.x and b.y > head.y:
                        dst = min(dst, (head.y - b.y) ** 2)
            #bottom-left
            if d.x < 0 and d.y > 0:
                for b in self.snake.body[1:]:
                    if b.y + b.x == head.y + head.x and b.y > head.y:
                        dst = min(dst, (head.y - b.y) ** 2 + (head.x - b.x) ** 2)

            return 1 / (1 + math.sqrt(dst))

        def apple_ang(dir: Point) -> float:
            if self.apple == head:
                return 0

            # tan(ang) = A x B / A dot B
            # A = head to apple vector
            # B = dir vector
            h2a = Point(self.apple.x - head.x, self.apple.y - head.y)
            crs_prod = h2a.x * dir.y - h2a.y * dir.x
            dot_prod = h2a.x * dir.x + h2a.y * dir.y
            return math.atan2(crs_prod, dot_prod)


        return ([

            wall_dist(dir),
            wall_dist(Point(dir.y, -dir.x)),
            wall_dist(Point(-dir.y, dir.x)),

            body_dist(dir),  #f
            body_dist(Point(dir.y, -dir.x)),  #l
            body_dist(Point(-dir.y, dir.x)),  #r

            body_dist(Point(dir.y + dir.x, dir.y - dir.x)),  #forward-left
            body_dist(Point(dir.y - dir.x, -dir.y - dir.x)),  #backward-left
            body_dist(Point(-dir.y - dir.x, -dir.y + dir.x)),  #backward-right
            body_dist(Point(-dir.y + dir.x, dir.y + dir.x)),

            apple_ang(dir) / math.pi,
        ])
