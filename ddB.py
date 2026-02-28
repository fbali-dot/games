"""
The Forest Guardian - A Grid-Based Puzzle Game

A Guardian must push seeds onto fertile soil tiles to grow trees
while navigating obstacles like rocks and water.

Architecture:
- Entity (base class) -> Guardian, Seed
- Tile types: Grass, Soil, Rock, Water
- LevelLoader: Parses map data from string arrays
- GameUI: Tracks moves and current level
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Set
import copy


# =============================================================================
# COLOR PALETTE
# =============================================================================

class Colors:
    """Color palette for the game."""
    GREEN = (34, 139, 34)      # Player (Forest Green)
    BROWN = (139, 69, 19)      # Seeds (Saddle Brown)
    YELLOW = (218, 165, 32)    # Soil (Goldenrod)
    BLUE = (65, 105, 225)      # Water (Royal Blue)
    GRAY = (105, 105, 105)     # Rocks (Dim Gray)
    LIGHT_GREEN = (144, 238, 144)  # Grass (Light Green)
    DARK_GREEN = (0, 100, 0)   # Trees (Dark Green)
    WHITE = (255, 255, 255)    # UI Text
    BLACK = (0, 0, 0)          # Background/UI
    GOLD = (255, 215, 0)       # Highlight


# =============================================================================
# DIRECTION
# =============================================================================

class Direction(Enum):
    """Movement directions."""
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    
    @property
    def delta(self) -> Tuple[int, int]:
        """Return the (dx, dy) movement delta."""
        deltas = {
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0)
        }
        return deltas[self]


# =============================================================================
# POSITION
# =============================================================================

@dataclass(frozen=True)
class Position:
    """Immutable position on the grid."""
    x: int
    y: int
    
    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Position') -> 'Position':
        return Position(self.x - other.x, self.y - other.y)
    
    def move(self, direction: Direction) -> 'Position':
        """Move position in given direction."""
        dx, dy = direction.delta
        return Position(self.x + dx, self.y + dy)
    
    def as_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


# =============================================================================
# TILE TYPES
# =============================================================================

class TileType(Enum):
    """Types of tiles in the game grid."""
    GRASS = auto()
    SOIL = auto()
    ROCK = auto()
    WATER = auto()
    TREE = auto()  # Grown tree (seed on soil)


@dataclass
class Tile:
    """Represents a single tile on the grid."""
    tile_type: TileType
    position: Position
    
    @property
    def is_walkable(self) -> bool:
        """Check if entities can walk on this tile."""
        return self.tile_type in (TileType.GRASS, TileType.SOIL)
    
    @property
    def is_fertile(self) -> bool:
        """Check if this tile can grow a tree (soil)."""
        return self.tile_type == TileType.SOIL
    
    @property
    def is_obstacle(self) -> bool:
        """Check if this tile blocks movement."""
        return self.tile_type in (TileType.ROCK, TileType.WATER, TileType.TREE)
    
    def grow_tree(self) -> 'Tile':
        """Transform soil into a tree."""
        if self.tile_type == TileType.SOIL:
            return Tile(TileType.TREE, self.position)
        return self


# =============================================================================
# ENTITY BASE CLASS
# =============================================================================

class Entity(ABC):
    """
    Base class for all game entities.
    
    This shared base class provides common functionality for both
    the Guardian (player) and Seeds, including position management,
    movement, and rendering properties.
    """
    
    def __init__(self, position: Position, entity_id: int = 0):
        self._position = position
        self._entity_id = entity_id
        self._is_active = True
    
    @property
    def position(self) -> Position:
        """Current position of the entity."""
        return self._position
    
    @position.setter
    def position(self, new_position: Position) -> None:
        """Update the entity's position."""
        self._position = new_position
    
    @property
    def entity_id(self) -> int:
        """Unique identifier for this entity."""
        return self._entity_id
    
    @property
    def is_active(self) -> bool:
        """Whether this entity is active in the game."""
        return self._is_active
    
    @is_active.setter
    def is_active(self, value: bool) -> None:
        """Set entity active state."""
        self._is_active = value
    
    @property
    @abstractmethod
    def symbol(self) -> str:
        """Character symbol for text rendering."""
        pass
    
    @property
    @abstractmethod
    def color(self) -> Tuple[int, int, int]:
        """RGB color for graphical rendering."""
        pass
    
    @property
    @abstractmethod
    def can_be_pushed(self) -> bool:
        """Whether this entity can be pushed by the player."""
        pass
    
    def move_to(self, new_position: Position) -> None:
        """Move entity to a new position."""
        self._position = new_position
    
    def move(self, direction: Direction) -> Position:
        """
        Calculate new position after moving in a direction.
        Does not actually move - returns the target position.
        """
        return self._position.move(direction)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pos={self._position}, id={self._entity_id})"


# =============================================================================
# GUARDIAN (PLAYER)
# =============================================================================

class Guardian(Entity):
    """
    The Guardian is the player-controlled character.
    
    The Guardian moves around the grid and pushes seeds onto soil tiles.
    Color: Green (as specified)
    """
    
    def __init__(self, position: Position, entity_id: int = 0):
        super().__init__(position, entity_id)
        self._moves_count = 0
    
    @property
    def symbol(self) -> str:
        return 'G'
    
    @property
    def color(self) -> Tuple[int, int, int]:
        return Colors.GREEN
    
    @property
    def can_be_pushed(self) -> bool:
        return False  # Player cannot be pushed
    
    @property
    def moves_count(self) -> int:
        """Total number of moves made by the Guardian."""
        return self._moves_count
    
    def increment_moves(self) -> None:
        """Increment the move counter."""
        self._moves_count += 1
    
    def reset_moves(self) -> None:
        """Reset the move counter."""
        self._moves_count = 0


# =============================================================================
# SEED
# =============================================================================

class Seed(Entity):
    """
    Seeds are pushable objects that can be placed on soil to grow trees.
    
    When a seed is pushed onto a fertile soil tile, it transforms into a tree.
    Color: Brown (as specified)
    """
    
    def __init__(self, position: Position, entity_id: int = 0):
        super().__init__(position, entity_id)
        self._is_planted = False
    
    @property
    def symbol(self) -> str:
        return 'S' if not self._is_planted else 'T'
    
    @property
    def color(self) -> Tuple[int, int, int]:
        return Colors.BROWN
    
    @property
    def can_be_pushed(self) -> bool:
        return True  # Seeds can be pushed by the Guardian
    
    @property
    def is_planted(self) -> bool:
        """Whether this seed has been planted on soil."""
        return self._is_planted
    
    def plant(self) -> None:
        """Mark this seed as planted."""
        self._is_planted = True
        self._is_active = False  # Seed is no longer active after planting


# =============================================================================
# GAME GRID
# =============================================================================

class GameGrid:
    """
    The game grid manages all tiles and provides spatial queries.
    """
    
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self._tiles: Dict[Tuple[int, int], Tile] = {}
        self._initialize_grass()
    
    def _initialize_grass(self) -> None:
        """Fill the grid with grass tiles by default."""
        for y in range(self._height):
            for x in range(self._width):
                pos = Position(x, y)
                self._tiles[pos.as_tuple()] = Tile(TileType.GRASS, pos)
    
    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height
    
    def get_tile(self, position: Position) -> Optional[Tile]:
        """Get tile at position, or None if out of bounds."""
        return self._tiles.get(position.as_tuple())
    
    def set_tile(self, position: Position, tile_type: TileType) -> None:
        """Set a tile type at the given position."""
        if 0 <= position.x < self._width and 0 <= position.y < self._height:
            self._tiles[position.as_tuple()] = Tile(tile_type, position)
    
    def is_valid_position(self, position: Position) -> bool:
        """Check if position is within grid bounds."""
        return (0 <= position.x < self._width and 
                0 <= position.y < self._height)
    
    def is_walkable(self, position: Position) -> bool:
        """Check if a position is walkable (for the Guardian)."""
        tile = self.get_tile(position)
        return tile is not None and tile.is_walkable
    
    def grow_tree_at(self, position: Position) -> bool:
        """Transform soil at position into a tree. Returns True if successful."""
        tile = self.get_tile(position)
        if tile and tile.is_fertile:
            self._tiles[position.as_tuple()] = tile.grow_tree()
            return True
        return False
    
    def get_all_tiles_of_type(self, tile_type: TileType) -> List[Tile]:
        """Get all tiles of a specific type."""
        return [t for t in self._tiles.values() if t.tile_type == tile_type]
    
    def count_tiles_of_type(self, tile_type: TileType) -> int:
        """Count tiles of a specific type."""
        return sum(1 for t in self._tiles.values() if t.tile_type == tile_type)


# =============================================================================
# LEVEL LOADER
# =============================================================================

class LevelLoader:
    """
    Parses level data from string-based arrays.
    
    Map Legend:
    - '.' or ' ' : Grass (walkable)
    - 'X' or '#' : Rock (obstacle)
    - '~' or 'W' : Water (obstacle)
    - 'O' or '+' : Soil (fertile, target for seeds)
    - 'G' or 'P' : Guardian starting position
    - 'S' or 'B' : Seed position
    - 'T' or '*' : Tree (already grown)
    
    The loader creates the grid, places entities, and tracks objectives.
    """
    
    # Character mappings for flexibility
    TILE_CHARS = {
        '.': TileType.GRASS,
        ' ': TileType.GRASS,
        'X': TileType.ROCK,
        '#': TileType.ROCK,
        '~': TileType.WATER,
        'W': TileType.WATER,
        'O': TileType.SOIL,
        '+': TileType.SOIL,
        'T': TileType.TREE,
        '*': TileType.TREE,
    }
    
    ENTITY_CHARS = {
        'G': 'guardian',
        'P': 'guardian',
        'S': 'seed',
        'B': 'seed',
    }
    
    @classmethod
    def load_from_strings(cls, level_data: List[str]) -> 'LoadedLevel':
        """
        Parse a level from an array of strings.
        
        Args:
            level_data: List of strings, each representing a row of the map.
        
        Returns:
            LoadedLevel containing grid, entities, and metadata.
        """
        if not level_data:
            raise ValueError("Level data cannot be empty")
        
        # Normalize line lengths
        max_width = max(len(line) for line in level_data)
        normalized = [line.ljust(max_width) for line in level_data]
        
        height = len(normalized)
        width = max_width
        
        grid = GameGrid(width, height)
        guardian: Optional[Guardian] = None
        seeds: List[Seed] = []
        soil_positions: List[Position] = []
        entity_id = 0
        
        for y, line in enumerate(normalized):
            for x, char in enumerate(line):
                pos = Position(x, y)
                
                # Check for tile type
                if char in cls.TILE_CHARS:
                    tile_type = cls.TILE_CHARS[char]
                    grid.set_tile(pos, tile_type)
                    if tile_type == TileType.SOIL:
                        soil_positions.append(pos)
                
                # Check for entity
                if char in cls.ENTITY_CHARS:
                    entity_type = cls.ENTITY_CHARS[char]
                    if entity_type == 'guardian':
                        guardian = Guardian(pos, entity_id)
                        entity_id += 1
                    elif entity_type == 'seed':
                        seeds.append(Seed(pos, entity_id))
                        entity_id += 1
                        # Seed sits on grass by default
                        grid.set_tile(pos, TileType.GRASS)
        
        if guardian is None:
            raise ValueError("Level must have a Guardian (G or P)")
        
        return LoadedLevel(
            grid=grid,
            guardian=guardian,
            seeds=seeds,
            soil_positions=soil_positions,
            total_soil=len(soil_positions)
        )
    
    @classmethod
    def load_multiple_levels(cls, levels: List[List[str]]) -> List['LoadedLevel']:
        """Load multiple levels from a list of level data arrays."""
        return [cls.load_from_strings(level_data) for level_data in levels]


@dataclass
class LoadedLevel:
    """Container for loaded level data."""
    grid: GameGrid
    guardian: Guardian
    seeds: List[Seed]
    soil_positions: List[Position]
    total_soil: int
    
    @property
    def seed_count(self) -> int:
        return len(self.seeds)


# =============================================================================
# GAME UI
# =============================================================================

class GameUI:
    """
    Screen interface for tracking game state.
    
    Displays:
    - Number of moves made
    - Current level number
    - Number of seeds remaining
    - Number of trees grown
    """
    
    def __init__(self):
        self._moves: int = 0
        self._current_level: int = 1
        self._seeds_remaining: int = 0
        self._trees_grown: int = 0
        self._total_trees_needed: int = 0
        self._message: str = ""
        self._level_complete: bool = False
    
    @property
    def moves(self) -> int:
        return self._moves
    
    @moves.setter
    def moves(self, value: int) -> None:
        self._moves = max(0, value)
    
    @property
    def current_level(self) -> int:
        return self._current_level
    
    @current_level.setter
    def current_level(self, value: int) -> None:
        self._current_level = max(1, value)
    
    @property
    def seeds_remaining(self) -> int:
        return self._seeds_remaining
    
    @property
    def trees_grown(self) -> int:
        return self._trees_grown
    
    @property
    def total_trees_needed(self) -> int:
        return self._total_trees_needed
    
    @property
    def message(self) -> str:
        return self._message
    
    @property
    def level_complete(self) -> bool:
        return self._level_complete
    
    def reset_for_level(self, level_num: int, total_seeds: int, total_soil: int) -> None:
        """Reset UI state for a new level."""
        self._moves = 0
        self._current_level = level_num
        self._seeds_remaining = total_seeds
        self._trees_grown = 0
        self._total_trees_needed = min(total_seeds, total_soil)
        self._message = ""
        self._level_complete = False
    
    def update_moves(self, moves: int) -> None:
        """Update the move counter display."""
        self._moves = moves
    
    def seed_planted(self) -> None:
        """Called when a seed is planted on soil."""
        self._seeds_remaining -= 1
        self._trees_grown += 1
        self._message = "A tree has grown!"
    
    def set_message(self, message: str) -> None:
        """Set a status message."""
        self._message = message
    
    def mark_level_complete(self) -> None:
        """Mark the current level as complete."""
        self._level_complete = True
        self._message = f"Level {self._current_level} Complete!"
    
    def render_text(self) -> str:
        """Render the UI as text."""
        lines = [
            "=" * 40,
            "       THE FOREST GUARDIAN",
            "=" * 40,
            f"  Level: {self._current_level}",
            f"  Moves: {self._moves}",
            f"  Seeds Remaining: {self._seeds_remaining}",
            f"  Trees Grown: {self._trees_grown}/{self._total_trees_needed}",
            "-" * 40,
        ]
        
        if self._message:
            lines.append(f"  >> {self._message}")
            lines.append("-" * 40)
        
        return "\n".join(lines)
    
    def get_status_dict(self) -> Dict:
        """Get UI state as a dictionary for external rendering."""
        return {
            'level': self._current_level,
            'moves': self._moves,
            'seeds_remaining': self._seeds_remaining,
            'trees_grown': self._trees_grown,
            'trees_needed': self._total_trees_needed,
            'message': self._message,
            'level_complete': self._level_complete
        }


# =============================================================================
# GAME STATE
# =============================================================================

class GameState:
    """
    Manages the complete game state including grid, entities, and logic.
    """
    
    def __init__(self, loaded_level: LoadedLevel, level_number: int = 1):
        self._grid = loaded_level.grid
        self._guardian = loaded_level.guardian
        self._seeds: Dict[Tuple[int, int], Seed] = {
            seed.position.as_tuple(): seed for seed in loaded_level.seeds
        }
        self._soil_positions = set(loaded_level.soil_positions)
        self._ui = GameUI()
        self._ui.reset_for_level(
            level_number,
            len(loaded_level.seeds),
            loaded_level.total_soil
        )
        self._trees_grown = 0
        self._is_won = False
    
    @property
    def grid(self) -> GameGrid:
        return self._grid
    
    @property
    def guardian(self) -> Guardian:
        return self._guardian
    
    @property
    def seeds(self) -> List[Seed]:
        return list(self._seeds.values())
    
    @property
    def ui(self) -> GameUI:
        return self._ui
    
    @property
    def is_won(self) -> bool:
        return self._is_won
    
    def get_seed_at(self, position: Position) -> Optional[Seed]:
        """Get seed at position if any."""
        return self._seeds.get(position.as_tuple())
    
    def has_seed_at(self, position: Position) -> bool:
        """Check if there's a seed at position."""
        return position.as_tuple() in self._seeds
    
    def move_guardian(self, direction: Direction) -> bool:
        """
        Attempt to move the Guardian in the given direction.
        
        Returns True if the move was successful.
        Handles pushing seeds and planting them on soil.
        """
        if self._is_won:
            return False
        
        current_pos = self._guardian.position
        target_pos = self._guardian.move(direction)
        
        # Check bounds
        if not self._grid.is_valid_position(target_pos):
            self._ui.set_message("Cannot move there!")
            return False
        
        # Check tile walkability
        target_tile = self._grid.get_tile(target_pos)
        if target_tile is None or target_tile.is_obstacle:
            self._ui.set_message("Blocked by obstacle!")
            return False
        
        # Check for seed at target position
        seed_at_target = self.get_seed_at(target_pos)
        
        if seed_at_target:
            # Try to push the seed
            push_target = target_pos.move(direction)
            
            # Check if we can push the seed
            if not self._can_push_seed_to(push_target):
                self._ui.set_message("Cannot push seed there!")
                return False
            
            # Push the seed
            self._push_seed(seed_at_target, push_target)
        
        # Move the guardian
        self._guardian.move_to(target_pos)
        self._guardian.increment_moves()
        self._ui.update_moves(self._guardian.moves_count)
        
        # Clear message on successful move
        if not self._ui.message.startswith("A tree"):
            self._ui.set_message("")
        
        # Check win condition
        self._check_win_condition()
        
        return True
    
    def _can_push_seed_to(self, position: Position) -> bool:
        """Check if a seed can be pushed to the given position."""
        # Must be valid position
        if not self._grid.is_valid_position(position):
            return False
        
        # Check tile
        tile = self._grid.get_tile(position)
        if tile is None:
            return False
        
        # Cannot push onto rocks, water, or existing trees
        if tile.tile_type in (TileType.ROCK, TileType.WATER, TileType.TREE):
            return False
        
        # Cannot push onto another seed
        if self.has_seed_at(position):
            return False
        
        return True
    
    def _push_seed(self, seed: Seed, new_position: Position) -> None:
        """Push a seed to a new position, planting it if on soil."""
        old_pos = seed.position
        
        # Remove from old position
        del self._seeds[old_pos.as_tuple()]
        
        # Move seed
        seed.move_to(new_position)
        
        # Check if landing on soil
        tile = self._grid.get_tile(new_position)
        if tile and tile.is_fertile:
            # Plant the seed - grow a tree
            seed.plant()
            self._grid.grow_tree_at(new_position)
            self._trees_grown += 1
            self._ui.seed_planted()
            
            # Remove from soil positions tracking
            self._soil_positions.discard(new_position)
        else:
            # Add to new position
            self._seeds[new_position.as_tuple()] = seed
    
    def _check_win_condition(self) -> None:
        """Check if all seeds have been planted (or enough to win)."""
        # Win when no more active seeds
        active_seeds = sum(1 for s in self._seeds.values() if s.is_active)
        
        if active_seeds == 0 and self._trees_grown > 0:
            self._is_won = True
            self._ui.mark_level_complete()
    
    def render_text(self) -> str:
        """Render the game state as text."""
        # Build the visual grid
        lines = []
        
        for y in range(self._grid.height):
            row = ""
            for x in range(self._grid.width):
                pos = Position(x, y)
                char = self._get_char_at(pos)
                row += char
            lines.append(row)
        
        # Combine UI and grid
        ui_text = self._ui.render_text()
        grid_text = "\n".join(lines)
        
        return f"{ui_text}\n{grid_text}\n"
    
    def _get_char_at(self, pos: Position) -> str:
        """Get the character to display at a position."""
        # Check for guardian first
        if self._guardian.position == pos:
            return self._guardian.symbol
        
        # Check for seed
        seed = self.get_seed_at(pos)
        if seed and seed.is_active:
            return seed.symbol
        
        # Check tile
        tile = self._grid.get_tile(pos)
        if tile:
            return self._get_tile_char(tile)
        
        return '?'
    
    def _get_tile_char(self, tile: Tile) -> str:
        """Get character for a tile type."""
        chars = {
            TileType.GRASS: '.',
            TileType.SOIL: 'O',
            TileType.ROCK: 'X',
            TileType.WATER: '~',
            TileType.TREE: 'T',
        }
        return chars.get(tile.tile_type, '?')


# =============================================================================
# GAME MANAGER
# =============================================================================

class GameManager:
    """
    Manages multiple levels and game progression.
    """
    
    def __init__(self, levels: List[List[str]]):
        self._levels = LevelLoader.load_multiple_levels(levels)
        self._current_level_idx = 0
        self._state: Optional[GameState] = None
        self._total_moves = 0
        self._load_current_level()
    
    def _load_current_level(self) -> None:
        """Load the current level into a game state."""
        if 0 <= self._current_level_idx < len(self._levels):
            level = self._levels[self._current_level_idx]
            self._state = GameState(level, self._current_level_idx + 1)
    
    @property
    def state(self) -> Optional[GameState]:
        return self._state
    
    @property
    def current_level_number(self) -> int:
        return self._current_level_idx + 1
    
    @property
    def total_levels(self) -> int:
        return len(self._levels)
    
    @property
    def is_game_complete(self) -> bool:
        """Check if all levels are completed."""
        return self._current_level_idx >= len(self._levels)
    
    def move(self, direction: Direction) -> bool:
        """Make a move in the current level."""
        if self._state:
            return self._state.move_guardian(direction)
        return False
    
    def next_level(self) -> bool:
        """
        Advance to the next level.
        
        Returns True if there is a next level, False if game is complete.
        """
        if self._state:
            self._total_moves += self._state.guardian.moves_count
        
        self._current_level_idx += 1
        
        if self._current_level_idx < len(self._levels):
            self._load_current_level()
            return True
        
        self._state = None
        return False
    
    def restart_level(self) -> None:
        """Restart the current level."""
        self._load_current_level()
    
    def render_text(self) -> str:
        """Render current game state as text."""
        if self._state:
            return self._state.render_text()
        return "Game Complete!\n"


# =============================================================================
# SAMPLE LEVELS
# =============================================================================

SAMPLE_LEVELS = [
    # Level 1 - Simple introduction
    [
        "XXXXXXXXXX",
        "X........X",
        "X.G..S...X",
        "X........X",
        "X....O...X",
        "X........X",
        "XXXXXXXXXX",
    ],
    # Level 2 - Two seeds
    [
        "XXXXXXXXXXXX",
        "X..........X",
        "X.S.....S..X",
        "X..........X",
        "X....G.....X",
        "X..........X",
        "X..O....O..X",
        "X..........X",
        "XXXXXXXXXXXX",
    ],
    # Level 3 - With water obstacle
    [
        "XXXXXXXXXXXX",
        "X..........X",
        "X.G........X",
        "X..........X",
        "X...S..S....X",
        "X...~~~~....X",
        "X...~~~~....X",
        "X..O....O...X",
        "X..........X",
        "XXXXXXXXXXXX",
    ],
    # Level 4 - Rock maze
    [
        "XXXXXXXXXXXXX",
        "X.....X.....X",
        "X.S.X.X.X.S.X",
        "X...X...X...X",
        "XXXXX.G.XXXXX",
        "X...........X",
        "X.X.X.O.X.X.X",
        "X.X.X.O.X.X.X",
        "X...........X",
        "XXXXXXXXXXXXX",
    ],
    # Level 5 - Complex puzzle
    [
        "XXXXXXXXXXXXXXX",
        "X.............X",
        "X.S.X...X.X.S.X",
        "X...X...X.X...X",
        "X...X.G.X.X...X",
        "X...X...X.X...X",
        "X...X...X.X...X",
        "X..O~...~O....X",
        "X..O~...~O....X",
        "X.............X",
        "XXXXXXXXXXXXXXX",
    ],
]


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def print_instructions():
    """Print game instructions."""
    print("""
╔════════════════════════════════════════════════════════════╗
║              THE FOREST GUARDIAN - Instructions             ║
╠════════════════════════════════════════════════════════════╣
║  You are the Guardian (G). Push seeds (S) onto fertile     ║
║  soil (O) to grow trees (T). Avoid rocks (X) and water (~) ║
║                                                             ║
║  Controls:                                                  ║
║    W / ↑ : Move Up        S / ↓ : Move Down                 ║
║    A / ← : Move Left      D / → : Move Right                ║
║    R      : Restart Level  N : Next Level (after winning)   ║
║    Q      : Quit                                           ║
║                                                             ║
║  Symbols:                                                   ║
║    G = Guardian (You)     S = Seed (Push onto soil)         ║
║    O = Fertile Soil       T = Tree (Grown)                  ║
║    X = Rock               ~ = Water                         ║
║    . = Grass (walkable)                                    ║
╚════════════════════════════════════════════════════════════╝
""")


def play_game():
    """Main game loop for terminal play."""
    import sys
    
    print_instructions()
    
    game = GameManager(SAMPLE_LEVELS)
    
    # Map key to direction
    key_map = {
        'w': Direction.UP, 'W': Direction.UP,
        's': Direction.DOWN, 'S': Direction.DOWN,
        'a': Direction.LEFT, 'A': Direction.LEFT,
        'd': Direction.RIGHT, 'D': Direction.RIGHT,
        '\x1b[A': Direction.UP,     # Arrow up
        '\x1b[B': Direction.DOWN,   # Arrow down
        '\x1b[C': Direction.RIGHT,  # Arrow right
        '\x1b[D': Direction.LEFT,   # Arrow left
    }
    
    while not game.is_game_complete:
        print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
        print(game.render_text())
        
        if game.state and game.state.is_won:
            print("\n  🌳 Level Complete! Press N for next level or R to restart. 🌳")
        
        try:
            # Read single character
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                char = sys.stdin.read(1)
                # Handle escape sequences (arrow keys)
                if char == '\x1b':
                    char += sys.stdin.read(2)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            
            if char in ('q', 'Q'):
                print("\nThanks for playing The Forest Guardian!")
                break
            elif char in ('r', 'R'):
                game.restart_level()
            elif char in ('n', 'N'):
                if game.state and game.state.is_won:
                    if not game.next_level():
                        print("\n🎉 Congratulations! You completed all levels! 🎉")
                        break
            elif char in key_map:
                game.move(key_map[char])
        except (ImportError, KeyboardInterrupt):
            # Fallback for non-terminal environments
            char = input("\nEnter command (W/A/S/D to move, R to restart, N for next, Q to quit): ").strip()
            if char.lower() == 'q':
                print("Thanks for playing The Forest Guardian!")
                break
            elif char.lower() == 'r':
                game.restart_level()
            elif char.lower() == 'n':
                if game.state and game.state.is_won:
                    if not game.next_level():
                        print("\n🎉 Congratulations! You completed all levels! 🎉")
                        break
            elif char in key_map:
                game.move(key_map[char])


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    play_game()