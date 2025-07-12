# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 22:14:35 2021

@author: L
"""
import numpy as np
import heapq


class Board(object):
    """
    data: NxN的二维数组，list or narray; 例如: np.array([[0,1,3],[4,2,5],[7,8,6]])
    """

    def __init__(self, data, use_func="manhattan"):
        self.board_data = data
        self.n = len(self.board_data)
        self.goal = "_".join([str(i) for i in range(1, self.n * self.n)]) + "_0"
        self.prev = None
        self.height = 0
        self.use_func = use_func

    # board size n
    def size(self):
        return self.n

    # 从val值，得到其在八数码中的正确的问题。
    def get_xy_from_value(self, val):
        idx = (val - 1) // self.n
        idy = (val - 1) - idx * self.n
        return idx, idy

    # 当前状态到目标状态的海明距离
    def hamming(self):
        dist = 0
        for i in range(self.n):
            for j in range(self.n):
                if i == self.n - 1 and j == self.n - 1:
                    break
                goal = i * self.n + j + 1
                dist += abs(self.board_data[i][j] - goal)
        return dist

    # 当前状态到目标状态的曼哈顿距离
    def manhattan(self):
        dist = 0
        for i in range(self.n):
            for j in range(self.n):
                if i == self.n - 1 and j == self.n - 1:
                    break
                val = self.board_data[i][j]
                idx, idy = self.get_xy_from_value(val)
                dist += abs(i - idx) + abs(j - idy)
        return dist

    # 启发式的得分 source
    def get_score(self):
        if self.use_func == "hamming":
            return self.height + self.hamming()
        return self.height + self.manhattan()

    # 是否是目标状态
    def is_target(self):
        pass

    # 当前状态的所有邻居节点，return list<Board>
    def neighbors(self):
        all_neighbors = []
        return all_neighbors

    # 当前状态是否是最终状态
    def is_solvable(self):
        self.string = self.to_string()
        if self.string == self.goal:
            return True
        return False

    # 棋牌状态的string表示，例如return "1 2 3 4 5 6 7 8 0"
    def to_string(self):
        res = []
        for i in range(self.n):
            for j in range(self.n):
                res.append(str(self.board_data[i][j]))
        return "_".join(res)

    # 使用数值表示当前状态的唯一性，用于迭代过程中的判断
    def hash_code(self):
        hash_value = 0
        for i in range(self.n):
            for j in range(self.n):
                hash_value = hash_value * 10 + self.board_data[i][j]
        return hash_value

    def findzero(self):
        # 找到数字0
        for row in range(self.n):
            for col in range(self.n):
                if self.board_data[row][col] == 0:
                    break
            if self.board_data[row][col] == 0:
                break
        return row, col

    def __lt__(self, other):
        return other.get_score() > self.get_score()

    def __le__(self, other):
        return other.get_score() >= self.get_score()

    def __gt__(self, other):
        return other.get_score() < self.get_score()

    def __ge__(self, other):
        return other.get_score() <= self.get_score()

    def calInversionNumber(self):
        nums = list(self.board_data.reshape(self.n * self.n))
        for i in range(len(nums)):
            if nums[i] == 0:
                del nums[i]
                break
        # 归并统计逆序对
        tmp = nums.copy()

        def sortlist(l, r):
            if l >= r:
                return 0
            mid = l + (r - l) // 2
            res = sortlist(l, mid) + sortlist(mid + 1, r)
            # 合并
            # 剪枝
            if nums[mid] < nums[mid + 1]:
                return res
            tmp[l : r + 1] = nums[l : r + 1]
            i, j = l, mid + 1
            for k in range(l, r + 1):
                if i == mid + 1:
                    nums[k] = tmp[j]
                    j += 1
                elif j == r + 1 or tmp[i] <= tmp[j]:
                    nums[k] = nums[i]
                    i += 1
                else:
                    nums[k] = tmp[j]
                    j += 1
                    res += mid - i + 1
            return res

        res = sortlist(0, len(nums) - 1)
        if self.n & 1 == 1 and res & 1 == 0:
            return True  # 奇数且逆序对为偶数
        row, col = self.findzero()
        if self.n & 1 == 0 and (res + row) & 1 == 1:
            return True
        return False


from copy import deepcopy


# 这里是解决八数码问题的函数，本次测试都是可解初始化状态，对验证当前状态会否可解不做要求
class Solver(object):
    """
    board：初始化棋盘状态。
    use_algo: 解决八数码问题的算法，bfs和astart，当前你也可以实现dfs

    """

    def __init__(self, board, use_algo="bfs"):
        self.board = board
        self.ans_board = None
        self.step = -1
        self.use_algo = use_algo

    # 解决八数码问题，你必须实验这个函数，用来解决八数码问题
    def solver(self):
        if self.board.calInversionNumber():
            if self.use_algo == "bfs":
                self.solver_bfs()
            else:
                self.solver_astar()
            return self.ans_board
        else:
            return -1

    def swap(self, board, row1, col1, row2, col2):
        board[row1][col1], board[row2][col2] = board[row2][col2], board[row1][col1]

    def solver_bfs(self):
        step = 0
        queue = [self.board]
        dxdy = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        visited = set()
        visited.add(self.board.to_string())
        n = self.board.n
        while queue:
            size = len(queue)
            for i in range(size):
                cur = queue.pop(0)
                if cur.is_solvable():
                    self.step = step
                    self.ans_board = cur
                    return
                # 找到0
                row, col = cur.findzero()
                for dx, dy in dxdy:
                    x, y = row + dx, col + dy
                    if 0 <= x < n and 0 <= y < n:
                        new_board = deepcopy(cur)
                        self.swap(new_board.board_data, x, y, row, col)
                        new_board.prev = cur
                        new_str = new_board.to_string()
                        if new_str not in visited:
                            queue.append(new_board)
                            visited.add(new_str)
                        else:
                            del new_board
            step += 1
        return -1

    def solver_astar(self):
        queue = []  # 优先级队列
        heapq.heappush(queue, self.board)
        dxdy = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        visited = set()
        visited.add(self.board.to_string())
        n = self.board.n
        while queue:
            cur = heapq.heappop(queue)
            if cur.is_solvable():
                self.ans_board = cur
                self.step = self.ans_board.height
                return
            # 找到0
            row, col = cur.findzero()
            for dx, dy in dxdy:
                x, y = row + dx, col + dy
                if 0 <= x < n and 0 <= y < n:
                    new_board = deepcopy(cur)
                    self.swap(new_board.board_data, x, y, row, col)
                    new_board.prev = cur
                    new_str = new_board.to_string()
                    new_board.height += 1
                    if new_str not in visited:
                        heapq.heappush(queue, new_board)
                        visited.add(new_str)
                    else:
                        del new_board
        return -1

    # 返回最短的路径
    def moves(self):
        return self.step

    # 返回最优结果的路径，你必须返回这个结果
    def solution(self):
        if self.ans_board:
            prev = self.ans_board
            ans_list = [prev.board_data]
            while prev.prev:
                prev = prev.prev
                ans_list.append(prev.board_data)
            return ans_list[::-1]
        else:
            return []


def test_puzzle_4():
    n = 4
    random_list = [1, 6, 2, 4, 5, 0, 3, 8, 9, 10, 7, 11, 13, 14, 15, 12]
    board_data = np.array(random_list).reshape(n, n)
    board = Board(board_data)

    solver = Solver(board)
    solver.solver()

    move = solver.moves()
    solution = solver.solution()
    print(move)
    for s in solution:
        print(s)

    solver = Solver(board, "bfs")
    solver.solver()

    move = solver.moves()
    solution = solver.solution()
    print(move)
    for s in solution:
        print(s)


def test_puzzle_3():
    n = 3
    random_list = [0, 1, 3, 4, 2, 5, 7, 8, 6]
    board_data = np.array(random_list).reshape(n, n)
    board = Board(board_data)

    solver = Solver(board)
    solver.solver()

    move = solver.moves()
    solution = solver.solution()
    print(move)
    for s in solution:
        print(s)

    solver = Solver(board, "bfs")
    solver.solver()

    move = solver.moves()
    solution = solver.solution()
    print(move)
    for s in solution:
        print(s)


if __name__ == "__main__":
    import glob

    filelist = glob.glob(
        "D:\\Study_Work\\Electronic_data\\CS\\AAAUniversity\\rgznjc\\Lab\\L1\\Algorithm-hands-on-main\\8puzzle\\8puzzle\\puzzle3x3*.txt"
    )
    for i in range(len(filelist)):
        file = filelist[i]
        print(str(i) + " " + file, end=" ")
        f = open(file)
        data = f.readlines()
        data = [d.replace("\n", "") for d in data]
        n = int(data[0])
        array = []
        for i in range(1, n + 1):
            row = list(map(int, data[i].split()))
            array.append(row)
        board = Board(np.array(array))
        solver = Solver(board)  # ,'bfs'
        solver.solver()

        move = solver.moves()
        print("move: " + str(move) + " ", end=" ")
        solution = solver.solution()
        print("solution: " + str(len(solution)))
