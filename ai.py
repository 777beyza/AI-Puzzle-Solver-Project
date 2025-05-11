import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import random
import time
import heapq
from collections import deque
import threading

# Class representing puzzle situation
class PuzzleState:
    def __init__(self, board, parent=None, move=None, depth=30, size=3):
        self.board = board                    
        self.size = size                      
        self.parent = parent                  
        self.move = move                      
        self.depth = depth                    
        self.zero_pos = self.find_zero()      
        self.cost = 0                         

# Find the location of the empty cell
    def find_zero(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return (i, j)

    def __eq__(self, other):
        return self.board == other.board

    def __lt__(self, other):
        return self.cost < other.cost

    def __hash__(self):
        return hash(str(self.board))

# Function to check if the board is solvable or not
def is_solvable(board_flat, size):
    inv_count = sum(
        1 for i in range(len(board_flat)) for j in range(i + 1, len(board_flat))
        if board_flat[i] > board_flat[j] != 0 and board_flat[i] != 0
    )
    if size % 2 == 1:
        return inv_count % 2 == 0
    else:
        row_of_zero = (board_flat.index(0)) // size
        if (size - row_of_zero) % 2 == 1:
            return inv_count % 2 == 0
        else:
            return inv_count % 2 == 1

# Generate valid moves after the given state
def successors(state):
    next_states = []
    row, col = state.zero_pos
    size = state.size
    directions = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    for move, (dr, dc) in directions.items():
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < size and 0 <= new_col < size:
            new_board = [r[:] for r in state.board]
            new_board[row][col], new_board[new_row][new_col] = new_board[new_row][new_col], new_board[row][col]
            next_states.append(PuzzleState(new_board, parent=state, move=move, depth=state.depth + 1, size=size))
    return next_states

# Breadth-First Search algorithm
def bfs(start_state, goal_board):
    start_time = time.time()
    visited = set([start_state])
    queue = deque([start_state])
    while queue:
        current = queue.popleft()
        if current.board == goal_board:
            print("Number of nodes expanded:", len(visited))
            return {"solution": current, "expanded": len(visited), "moves": current.depth, "time": time.time() - start_time}
        for n in successors(current):
            if n not in visited:
                visited.add(n)
                queue.append(n)
    print("Number of nodes expanded:", len(visited))
    return None


# Depth-First Search algorithm
def dfs(start_state, goal_board, max_depth=100):
    start_time = time.time()
    visited = set()
    stack = [start_state]
    while stack:
        current = stack.pop()
        if current.board == goal_board:
            print("Number of nodes expanded:", len(visited))
            return {"solution": current, "expanded": len(visited), "moves": current.depth, "time": time.time() - start_time}
        if current in visited or current.depth > max_depth:
            continue
        visited.add(current)
        for n in reversed(successors(current)):
            if n not in visited:
                stack.append(n)
    print("Number of nodes expanded:", len(visited))
    return None


# Depth-Limited Search 
def dls(state, goal_board, limit, visited):
    if state.board == goal_board:
        return state
    if state.depth >= limit:
        return None
    visited.add(state)
    for n in successors(state):
        if n not in visited:
            res = dls(n, goal_board, limit, visited)
            if res:
                return res
    return None

# Iterative Deepening Search algorithm
def ids(start_state, goal_board, max_limit=100):
    start_time = time.time()
    for depth in range(max_limit):
        visited = set()
        res = dls(start_state, goal_board, depth, visited)
        if res:
            print("Number of nodes expanded:", len(visited))
            return {"solution": res, "expanded": len(visited), "moves": res.depth, "time": time.time() - start_time}
    print("Number of nodes expanded:", len(visited))
    return None


# Heuristic: Number of boxes in wrong place
def heuristic_misplaced(state, goal_board):
    return sum(
        1 for i in range(state.size) for j in range(state.size)
        if state.board[i][j] != 0 and state.board[i][j] != goal_board[i][j]
    )

# Heuristic: Manhattan distance
def heuristic_manhattan(state, goal_board):
    positions = {goal_board[i][j]: (i, j) for i in range(state.size) for j in range(state.size)}
    return sum(
        abs(i - positions[v][0]) + abs(j - positions[v][1])
        for i in range(state.size) for j in range(state.size)
        if (v := state.board[i][j]) != 0
    )

# A* algorithm
def a_star(start_state, goal_board, heuristic_fn):
    start_time = time.time()
    open_list = []
    visited = set()
    start_state.cost = heuristic_fn(start_state, goal_board)
    heapq.heappush(open_list, (start_state.cost, start_state))
    while open_list:
        _, current = heapq.heappop(open_list)
        if current.board == goal_board:
            print("Number of nodes expanded:", len(visited))
            return {"solution": current, "expanded": len(visited), "moves": current.depth, "time": time.time() - start_time}
        visited.add(current)
        for n in successors(current):
            if n not in visited:
                n.cost = n.depth + heuristic_fn(n, goal_board)
                heapq.heappush(open_list, (n.cost, n))
    print("Number of nodes expanded:", len(visited))
    return None


# GUI Interface with Tkinter
class PuzzleGUI:
    def __init__(self, root, size=3):
        self.root = root
        self.size = size
        self.board_frame = tk.Frame(root)
        self.board_frame.pack()
        self.status = None
        self.solution_thread = None
        self.stop_flag = False
        self.manual_mode = False
        self.manual_zeros = set()

        self.state = self.generate_random_state()
        self.generate_board()
        self.create_controls()

    # Generate random and solvable initialization
    def generate_random_state(self):
        while True:
            nums = list(range(self.size * self.size))
            random.shuffle(nums)
            if is_solvable(nums, self.size):
                board = [nums[i:i + self.size] for i in range(0, len(nums), self.size)]
                return PuzzleState(board, size=self.size)

    
    def generate_board(self):
        for w in self.board_frame.winfo_children():
            w.destroy()
        for i in range(self.size):
            for j in range(self.size):
                v = self.state.board[i][j]
                txt = "" if v == -1 or (v == 0 and (i, j) not in self.manual_zeros) else str(v)
                btn = tk.Button(
                    self.board_frame,
                    text=txt,
                    width=4 if self.size >= 5 else 5,
                    height=2,
                    font=("Helvetica", 14 if self.size >= 5 else 18),
                    command=lambda row=i, col=j: self.manual_cell_click(row, col)
                )
                btn.grid(row=i, column=j, padx=1, pady=1)

    # Interface controls
    def create_controls(self):
        cf = tk.Frame(self.root)
        cf.pack(pady=10)
        tk.Button(cf, text="Random Start", command=self.randomize).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="Load From File", command=self.load_from_file).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="Manual Edit", command=self.enter_manual_mode).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="Next Step", command=self.next_step).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="Stop", command=self.stop_solution).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="BFS", command=lambda: self.run_solver(bfs)).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="DFS", command=lambda: self.run_solver(dfs)).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="IDS", command=lambda: self.run_solver(ids)).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="A* Misplaced", command=lambda: self.run_solver(a_star, heuristic_misplaced)).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="A* Manhattan", command=lambda: self.run_solver(a_star, heuristic_manhattan)).pack(side=tk.LEFT, padx=2)
        self.status = tk.Label(cf, text="Step: 0")
        self.status.pack(side=tk.LEFT, padx=2)

    # Switching to manual entry mode
    def enter_manual_mode(self):
        self.manual_mode = True
        self.manual_zeros = set()
        empty_board = [[-1 for _ in range(self.size)] for _ in range(self.size)]
        self.state = PuzzleState(empty_board, size=self.size)
        self.generate_board()
        self.status.config(text="Manual Entry Mode:")

    # Manually entering a number into a cell
    def manual_cell_click(self, row, col):
        if not self.manual_mode:
            return
        if self.state.board[row][col] != -1:
            return
        val = simpledialog.askinteger("Enter Number", f"{row}, {col} for 0-{self.size * self.size - 1} enter a number between:")
        if val is None:
            return
        if not (0 <= val < self.size * self.size):
            messagebox.showerror("Incorrect Login", f"Please with 0 {self.size * self.size - 1} enter a number between .")
            return
        used_vals = [v for r in self.state.board for v in r if v != -1]
        if val in used_vals:
            messagebox.showerror("Recurring", f"{val} has already been used in another cell.")
            return
        self.state.board[row][col] = val
        if val == 0:
            self.manual_zeros.add((row, col))
        flat_board = [v for r in self.state.board for v in r]
        if flat_board.count(-1) == 1 and len(set(flat_board) - {-1}) == self.size * self.size - 1:
            for i in range(self.size):
                for j in range(self.size):
                    if self.state.board[i][j] == -1:
                        self.state.board[i][j] = 0
                        self.state.zero_pos = (i, j)
                        break
            self.state.depth = 0
            self.state.parent = None
            self.manual_mode = False
            flat_board = [v for r in self.state.board for v in r]
            if is_solvable(flat_board, self.size):
                self.status.config(text="✔ Valid and solvable situation")
            else:
                self.status.config(text="An unsolvable situation")
        self.generate_board()

    # Load initial state from file
    def load_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                board = [list(map(int, line.split())) for line in lines]
            size = len(board)
            flat_board = [num for row in board for num in row]
            if not is_solvable(flat_board, size):
                messagebox.showerror("Invalid", "This situation cannot be resolved.")
                return
            self.size = size
            self.state = PuzzleState(board, size=size)
            self.generate_board()
            self.status.config(text="Step: 0")
        except Exception as e:
            messagebox.showerror("Error", f"File could not be read: {e}")

    # Generating random new starter
    def randomize(self):
        self.state = self.generate_random_state()
        self.generate_board()
        self.status.config(text="Step: 0")

    def next_step(self):
        ns = successors(self.state)
        if ns:
            self.state = ns[0]
            self.generate_board()
            self.status.config(text=f"Step: {self.state.depth}")
        else:
            self.status.config(text="no next step")

    def stop_solution(self):
        self.stop_flag = True

    def run_solver(self, solver_fn, heuristic_fn=None):
        def thread_fn():
            self.stop_flag = False
            goal = [[(i * self.size + j + 1) % (self.size * self.size) for j in range(self.size)] for i in range(self.size)]
            if solver_fn == a_star:
                res = solver_fn(self.state, goal, heuristic_fn)
            else:
                res = solver_fn(self.state, goal)
            if not res:
                self.status.config(text="No solution found")
                return
            path = []
            node = res["solution"]
            while node:
                path.append(node)
                node = node.parent
            path.reverse()
            for i, state in enumerate(path):
                if self.stop_flag:
                    self.status.config(text="Stopped")
                    return
                self.state = state
                self.generate_board()
                self.status.config(text=f"Step: {i}")
                time.sleep(0.5 if self.size <= 5 else 0.8)
            self.status.config(
                text=f"✔ {solver_fn.__name__.upper()} {res['moves']} adım, {round(res['time'], 3)} sn, {res['expanded']} node"
            )
        if self.solution_thread and self.solution_thread.is_alive():
            messagebox.showinfo("Warning", "There is already a solution in the works.")
            return
        self.solution_thread = threading.Thread(target=thread_fn)
        self.solution_thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    size = simpledialog.askinteger("Puzzle Size", "Enter puzzle size (3, 5 or 7)")
    if size not in [3, 5, 7]:
        messagebox.showerror("Incorrect Login", "Only 3, 5 or 7 are accepted!")
        exit()
    root.deiconify()
    root.title(f"{size}x{size} Puzzle AI - BFS, DFS, IDS, A*")
    PuzzleGUI(root, size=size)
    root.mainloop()
