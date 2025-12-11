import time
from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Any

# --- Cube State Representation ---
# A simplified 3x3x3 cube state. Each face is a 3x3 grid.
# The faces are represented by a single character (e.g., 'U' for Up, 'R' for Right).
# Colors are also single characters (e.g., 'W' for White, 'Y' for Yellow).
# State: Dict[Face_Name, List[Color_Char]] where the list has 9 elements.
# Example: {'U': ['W']*9, 'R': ['R']*9, ...}

FaceState = List[str]
CubeState = Dict[str, FaceState]

class RubiksCube:
    """Represents the 3x3x3 Rubik's Cube."""
    
    # Standard face ordering (for visual and move consistency)
    FACES = ['U', 'D', 'L', 'R', 'F', 'B'] 
    
    # Standard colors for a solved cube
    SOLVED_COLORS = {
        'U': 'W', 'D': 'Y', 'L': 'O', 'R': 'R', 'F': 'G', 'B': 'B'
    }

    def __init__(self, initial_state: CubeState):
        self._state: CubeState = initial_state
        self._observers: List['Observer'] = []

    def get_state(self) -> CubeState:
        return self._state

    # --- Observer Pattern: Subject Methods ---
    def attach(self, observer: 'Observer') -> None:
        """Attaches an observer to the cube."""
        self._observers.append(observer)

    def detach(self, observer: 'Observer') -> None:
        """Detaches an observer from the cube."""
        self._observers.remove(observer)

    def _notify(self, move_name: str) -> None:
        """Notifies all observers about a move."""
        for observer in self._observers:
            observer.update(self._state, move_name)

    # --- Move Execution ---
    def apply_move(self, move_name: str, transformation: Callable[[CubeState], CubeState], notify: bool = True) -> None:
        """Applies a transformation and notifies observers."""
        # In a real implementation, this would involve complex state mutation logic.
        # For the skeleton, we'll just simulate a change and notification.
        self._state = transformation(self._state)
        if notify:
            self._notify(move_name)
    
    def is_solved(self) -> bool:
        """Checks if the cube is in the solved state."""
        # A simple check: all 9 stickers on each face must be the same color as the center.
        for face, color in self.SOLVED_COLORS.items():
            if any(s != color for s in self._state[face]):
                return False
        return True

# --- Observer Pattern ---

class Observer(ABC):
    """Abstract base class for all observers."""
    @abstractmethod
    def update(self, state: CubeState, move_name: str) -> None:
        """Receive update from subject (the Cube)."""
        pass

class ConsoleView(Observer):
    """Concrete Observer that prints the cube state to the console."""
    
    def update(self, state: CubeState, move_name: str) -> None:
        """Prints the cube state after a move."""
        print(f"\n--- Move Executed: **{move_name}** ---")
        print("Cube State (Simplified):")
        # In a real app, this would format the output nicely (e.g., ASCII art)
        for face, face_state in state.items():
            print(f"  {face}: {face_state[4]} (Center Color) | First 3 Stickers: {face_state[:3]}")

# --- Command Pattern ---

class Command(ABC):
    """Base class for all cube rotation commands (moves)."""
    
    def __init__(self, cube: RubiksCube):
        self._cube = cube
        self._move_name = ""
        
    @abstractmethod
    def execute(self) -> None:
        """Performs the move on the cube."""
        pass

    @abstractmethod
    def undo(self) -> None:
        """Undoes the move on the cube (e.g., F -> F')."""
        pass
        
# A helper function to simulate a state change for the skeleton.
# In a real implementation, this would be the actual move logic.
def simulate_move_transformation(state: CubeState, move_char: str) -> CubeState:
    """Returns a copy of the state, simulating a change."""
    new_state = {face: list(stickers) for face, stickers in state.items()}
    # Dummy change for demonstration: just flip the center sticker color of the moved face
    # (This is NOT a real Rubik's Cube move logic)
    if move_char in new_state:
        # Toggle center color for 'scrambling' effect
        center_color = new_state[move_char][4]
        if center_color == 'W':
            new_state[move_char][4] = 'Y'
        else:
            new_state[move_char][4] = 'W'
    return new_state

class MoveF(Command):
    """Concrete Command for a clockwise Face (F) rotation."""
    
    def __init__(self, cube: RubiksCube):
        super().__init__(cube)
        self._move_name = "F"

    def execute(self) -> None:
        print(f"Executing: {self._move_name}")
        # Pass the actual move logic function (or lambda) to the cube
        self._cube.apply_move(self._move_name, lambda s: simulate_move_transformation(s, 'F'))

    def undo(self) -> None:
        print(f"Undoing: {self._move_name} (Executing F' move)")
        # In a real app, F' would have its own transformation logic
        self._cube.apply_move("F'", lambda s: simulate_move_transformation(s, 'F'), notify=False) # notify=False for clean undo log

class MoveR(Command):
    """Concrete Command for a clockwise Right (R) rotation."""
    
    def __init__(self, cube: RubiksCube):
        super().__init__(cube)
        self._move_name = "R"

    def execute(self) -> None:
        print(f"Executing: {self._move_name}")
        self._cube.apply_move(self._move_name, lambda s: simulate_move_transformation(s, 'R'))

    def undo(self) -> None:
        print(f"Undoing: {self._move_name} (Executing R' move)")
        self._cube.apply_move("R'", lambda s: simulate_move_transformation(s, 'R'), notify=False)

# --- Factory Pattern ---

class CubeFactory:
    """Factory class to create RubiksCube instances."""

    @staticmethod
    def _create_solved_state() -> CubeState:
        """Creates the state for a solved cube."""
        return {
            face: [color] * 9 for face, color in RubiksCube.SOLVED_COLORS.items()
        }

    @staticmethod
    def _create_random_state() -> CubeState:
        """Creates a 'random' (scrambled) state."""
        # For the skeleton, this will be a slightly perturbed version of the solved state
        solved_state = CubeFactory._create_solved_state()
        
        # Simple simulated scramble: swap two corner stickers on the Up face
        scrambled_state = {face: list(stickers) for face, stickers in solved_state.items()}
        u_stickers = scrambled_state['U']
        u_stickers[0], u_stickers[2] = u_stickers[2], u_stickers[0] # Swap U[0] and U[2]
        
        # In a real implementation, this would execute a long sequence of random, valid moves.
        return scrambled_state

    @staticmethod
    def create_cube(state_type: str) -> RubiksCube:
        """Initializes a RubiksCube based on the requested state type."""
        if state_type.lower() == "solved":
            initial_state = CubeFactory._create_solved_state()
        elif state_type.lower() == "random":
            initial_state = CubeFactory._create_random_state()
        else:
            raise ValueError(f"Unknown cube state type: {state_type}. Must be 'Solved' or 'Random'.")
        
        return RubiksCube(initial_state)

# --- Strategy Pattern ---

class SolverStrategy(ABC):
    """Abstract base class for all solving algorithms."""

    def __init__(self, cube: RubiksCube):
        self._cube = cube

    @abstractmethod
    def solve(self) -> bool:
        """Attempts to solve the cube using the implemented strategy."""
        pass

class LayerByLayerSolver(SolverStrategy):
    """Concrete Strategy: Solves the cube using the beginner's Layer-by-Layer method."""

    def solve(self) -> bool:
        """Executes the Layer-by-Layer solving algorithm."""
        
        print("\n" + "="*50)
        print("STARTING LAYER-BY-LAYER SOLVER STRATEGY")
        print("="*50)

        # 1. Solve the White Cross (First Layer)
        print("\n--- Phase 1: Solving the White Cross ---")
        # In a real solver, complex state analysis and move selection would occur here.
        # We will simulate the progress with key solving moves.
        
        # Example: Moves to get an edge piece into position
        cross_moves = [MoveR(self._cube), MoveF(self._cube), MoveF(self._cube)] 
        
        for i, move in enumerate(cross_moves):
            print(f"Progress: {i+1}/{len(cross_moves)} steps in White Cross...")
            move.execute()
            if self._cube.is_solved(): # Early exit check (unlikely here)
                print("Cube solved prematurely!")
                return True
        
        # 2. Solve the First Layer Corners
        print("\n--- Phase 2: Solving the First Layer Corners ---")
        # Example Corner Algorithm: R' D' R D (The "Sledgehammer")
        corner_alg = [
            MoveR(self._cube), MoveF(self._cube), MoveR(self._cube), MoveF(self._cube) 
        ]
        for i, move in enumerate(corner_alg):
             print(f"Progress: {i+1}/{len(corner_alg)} steps in First Layer Corners...")
             move.execute()
        
        # ... (Phases 3, 4, 5, 6, 7 would follow for a full Layer-by-Layer method) ...

        if self._cube.is_solved():
            print("\n*** CUBE SOLVED SUCCESSFULLY! ***")
            return True
        else:
            print("\n*** SOLVER FINISHED, CUBE MAY NOT BE FULLY SOLVED (Simplified Logic). ***")
            return False

# --- Main Execution Block ---

if __name__ == "__main__":
    
    print("--- Rubik's Cube Solver Application Start ---")
    
    # FACTORY PATTERN: Use the factory to create a cube in a random state.
    try:
        cube: RubiksCube = CubeFactory.create_cube("random")
        print(f"1. Cube initialized using Factory in 'Random' state.")
    except ValueError as e:
        print(f"Error initializing cube: {e}")
        exit()

    # OBSERVER PATTERN: Attach the ConsoleView observer.
    console_view = ConsoleView()
    cube.attach(console_view)
    print("2. ConsoleView Observer attached to the Cube.")
    print("   (Output will now show for all subsequent moves)")

    # COMMAND PATTERN: Manually execute a few Commands to scramble the cube further.
    print("\n3. Manually executing Commands (scrambling):")
    scramble_start_time = time.perf_counter()
    
    move_f = MoveF(cube)
    move_r = MoveR(cube)
    
    scramble_moves: List[Command] = [move_f, move_r, move_f, move_f, move_r]

    for move in scramble_moves:
        move.execute()
    
    scramble_duration = time.perf_counter() - scramble_start_time
    print(f"\nScramble sequence complete in {scramble_duration:.4f} seconds.")
    
    # STRATEGY PATTERN: Instantiate the Strategy solver and call .solve().
    print("\n4. Instantiating and executing LayerByLayerSolver (Strategy Pattern)...")
    solver: SolverStrategy = LayerByLayerSolver(cube)
    
    solve_start_time = time.perf_counter()
    is_solved = solver.solve()
    solve_duration = time.perf_counter() - solve_start_time
    
    # Output the result and time
    print("\n" + "#"*50)
    print(f"Solver run finished in {solve_duration:.4f} seconds.")
    print(f"Cube solved status: {'SUCCESS' if is_solved else 'FAILURE (Due to simplified logic)'}")
    print("#"*50)

    # Example of COMMAND UNDO (optional demonstration)
    print("\n5. Demonstrating Command UNDO:")
    move_f.undo()
    move_r.undo()
    print("Undo sequence complete.")