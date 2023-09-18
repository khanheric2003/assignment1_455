"""
gtp_connection.py
Module for playing games of Go using GoTextProtocol

Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller.
Parts of this code were originally based on the gtp module 
in the Deep-Go project by Isaac Henrion and Amos Storkey 
at the University of Edinburgh.
"""
import traceback
import numpy as np
import re
from sys import stdin, stdout, stderr
from typing import Any, Callable, Dict, List, Tuple

from board_base import (
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    GO_COLOR, GO_POINT,
    PASS,
    MAXSIZE,
    coord_to_point,
    opponent
)
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine


class GtpConnection:
    def __init__(self, go_engine: GoEngine, board: GoBoard, debug_mode: bool = False) -> None:
        """
        Manage a GTP connection for a Go-playing engine

        Parameters
        ----------
        go_engine:
            a program that can reply to a set of GTP commandsbelow
        board: 
            Represents the current board state.
        """
        self._debug_mode: bool = debug_mode
        self.go_engine = go_engine
        self.board: GoBoard = board
        self.commands: Dict[str, Callable[[List[str]], None]] = {
            "protocol_version": self.protocol_version_cmd,
            "quit": self.quit_cmd,
            "name": self.name_cmd,
            "boardsize": self.boardsize_cmd,
            "showboard": self.showboard_cmd,
            "clear_board": self.clear_board_cmd,
            "komi": self.komi_cmd,
            "version": self.version_cmd,
            "known_command": self.known_command_cmd,
            "genmove": self.genmove_cmd,
            "list_commands": self.list_commands_cmd,
            "play": self.play_cmd,
            "legal_moves": self.legal_moves_cmd,
            "gogui-rules_legal_moves": self.gogui_rules_legal_moves_cmd,
            "gogui-rules_final_result": self.gogui_rules_final_result_cmd,
            "gogui-rules_captured_count": self.gogui_rules_captured_count_cmd,
            "gogui-rules_game_id": self.gogui_rules_game_id_cmd,
            "gogui-rules_board_size": self.gogui_rules_board_size_cmd,
            "gogui-rules_side_to_move": self.gogui_rules_side_to_move_cmd,
            "gogui-rules_board": self.gogui_rules_board_cmd,
            "gogui-analyze_commands": self.gogui_analyze_cmd,
            "gogui-captured_check_commands": self.gogui_captured_check_cmd,
            "gogui-rules_legal_moves_cmd": self.gogui_rules_legal_moves_cmd_return,
            "gogui-test": self.gogui_test_cmd,
            "gogui-check_neighbors": self.gogui_check_neighbors_cmd

        }

        # argmap is used for argument checking
        # values: (required number of arguments,
        #          error message on argnum failure)
        self.argmap: Dict[str, Tuple[int, str]] = {
            "boardsize": (1, "Usage: boardsize INT"),
            "komi": (1, "Usage: komi FLOAT"),
            "known_command": (1, "Usage: known_command CMD_NAME"),
            "genmove": (1, "Usage: genmove {w,b}"),
            "play": (2, "Usage: play {b,w} MOVE"),
            "legal_moves": (1, "Usage: legal_moves {w,b}"),
        }

    def write(self, data: str) -> None:
        stdout.write(data)

    def flush(self) -> None:
        stdout.flush()

    def start_connection(self) -> None:
        """
        Start a GTP connection. 
        This function continuously monitors standard input for commands.
        """
        line = stdin.readline()
        while line:
            self.get_cmd(line)
            line = stdin.readline()

    def get_cmd(self, command: str) -> None:
        """
        Parse command string and execute it
        """
        if len(command.strip(" \r\t")) == 0:
            return
        if command[0] == "#":
            return
        # Strip leading numbers from regression tests
        if command[0].isdigit():
            command = re.sub("^\d+", "", command).lstrip()

        elements: List[str] = command.split()
        if not elements:
            return
        command_name: str = elements[0]
        args: List[str] = elements[1:]
        if self.has_arg_error(command_name, len(args)):
            return
        if command_name in self.commands:
            try:
                self.commands[command_name](args)
            except Exception as e:
                self.debug_msg("Error executing command {}\n".format(str(e)))
                self.debug_msg("Stack Trace:\n{}\n".format(
                    traceback.format_exc()))
                raise e
        else:
            self.debug_msg("Unknown command: {}\n".format(command_name))
            self.error("Unknown command")
            stdout.flush()

    def has_arg_error(self, cmd: str, argnum: int) -> bool:
        """
        Verify the number of arguments of cmd.
        argnum is the number of parsed arguments
        """
        if cmd in self.argmap and self.argmap[cmd][0] != argnum:
            self.error(self.argmap[cmd][1])
            return True
        return False

    def debug_msg(self, msg: str) -> None:
        """ Write msg to the debug stream """
        if self._debug_mode:
            stderr.write(msg)
            stderr.flush()

    def error(self, error_msg: str) -> None:
        """ Send error msg to stdout """
        stdout.write("? {}\n\n".format(error_msg))
        stdout.flush()

    def respond(self, response: str = "") -> None:
        """ Send response to stdout """
        stdout.write("= {}\n\n".format(response))
        stdout.flush()

    def reset(self, size: int) -> None:
        """
        Reset the board to empty board of given size
        """
        self.board.reset(size)

    def board2d(self) -> str:
        return str(GoBoardUtil.get_twoD_board(self.board))

    def protocol_version_cmd(self, args: List[str]) -> None:
        """ Return the GTP protocol version being used (always 2) """
        self.respond("2")

    def quit_cmd(self, args: List[str]) -> None:
        """ Quit game and exit the GTP interface """
        self.respond()
        exit()

    def name_cmd(self, args: List[str]) -> None:
        """ Return the name of the Go engine """
        self.respond(self.go_engine.name)

    def version_cmd(self, args: List[str]) -> None:
        """ Return the version of the  Go engine """
        self.respond(str(self.go_engine.version))

    def clear_board_cmd(self, args: List[str]) -> None:
        """ clear the board """
        self.reset(self.board.size)
        self.respond()

    def boardsize_cmd(self, args: List[str]) -> None:
        """
        Reset the game with new boardsize args[0]
        """
        self.reset(int(args[0]))
        self.respond()

    def showboard_cmd(self, args: List[str]) -> None:
        self.respond("\n" + self.board2d())

    def komi_cmd(self, args: List[str]) -> None:
        """
        Set the engine's komi to args[0]
        """
        self.go_engine.komi = float(args[0])
        self.respond()

    def known_command_cmd(self, args: List[str]) -> None:
        """
        Check if command args[0] is known to the GTP interface
        """
        if args[0] in self.commands:
            self.respond("true")
        else:
            self.respond("false")

    def list_commands_cmd(self, args: List[str]) -> None:
        """ list all supported GTP commands """
        self.respond(" ".join(list(self.commands.keys())))

    def legal_moves_cmd(self, args: List[str]) -> None:
        """
        List legal moves for color args[0] in {'b','w'}
        """
        board_color: str = args[0].lower()
        color: GO_COLOR = color_to_int(board_color)
        moves: List[GO_POINT] = GoBoardUtil.generate_legal_moves(
            self.board, color)
        gtp_moves: List[str] = []
        for move in moves:
            coords: Tuple[int, int] = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = " ".join(sorted(gtp_moves))
        self.respond(sorted_moves)

    """
    ==========================================================================
    Assignment 1 - game-specific commands start here
    ==========================================================================
    """
    """
    ==========================================================================
    Assignment 1 - commands we already implemented for you
    ==========================================================================
    """

    def gogui_analyze_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 1 """
        self.respond("pstring/Legal Moves For ToPlay/gogui-rules_legal_moves\n"
                     "pstring/Side to Play/gogui-rules_side_to_move\n"
                     "pstring/Final Result/gogui-rules_final_result\n"
                     "pstring/Board Size/gogui-rules_board_size\n"
                     "pstring/Rules GameID/gogui-rules_game_id\n"
                     "pstring/Show Board/gogui-rules_board\n"
                     )

    def gogui_rules_game_id_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 1 """
        self.respond("Ninuki")

    def gogui_rules_board_size_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 1 """
        self.respond(str(self.board.size))

    def gogui_rules_side_to_move_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 1 """
        color = "black" if self.board.current_player == BLACK else "white"
        self.respond(color)

    def gogui_rules_board_cmd(self, args: List[str]) -> None:
        """ We already implemented this function for Assignment 1 """
        size = self.board.size
        str = ''
        for row in range(size-1, -1, -1):
            start = self.board.row_start(row + 1)
            for i in range(size):
                # str += '.'
                point = self.board.board[start + i]
                if point == BLACK:
                    str += 'X'
                elif point == WHITE:
                    str += 'O'
                elif point == EMPTY:
                    str += '.'
                else:
                    assert False
            str += '\n'
        self.respond(str)

    """
    ==========================================================================
    Assignment 1 - game-specific commands you have to implement or modify
    ==========================================================================
    """

    def gogui_rules_final_result_cmd(self, args: List[str]) -> str:
        """ Implement this function for Assignment 1"""

        if self.check_5([]) == "white":
            self.respond("white")
            return "white"
        elif self.check_5([]) == "black":
            self.respond("black")
            return "black"
        elif len(self.board.get_empty_points()) == 0:
            self.respond("draw")
            return "draw"

        self.respond("unknown")
        return "unknown"

    def gogui_test_cmd(self, args: str) -> None:
        self.respond(str(args))

    def check_5(self, args: List[str]) -> str:

        size = self.board.size
        starting = 1
        for row in range(size - 1, -1, -1):
            start = self.board.row_start(row + 1)+size-1
            consecutive = 0
            last = EMPTY
            for i in range(starting):
                point = self.board.board[start + i * (size)]

                if point != last:
                    consecutive = 1
                    last = point
                elif point == last:
                    consecutive += 1
                if consecutive == 5 and last == WHITE:

                    return "white"
                elif consecutive == 5 and last == BLACK:

                    return "black"
            starting += 1

        size = self.board.size
        start = self.board.row_start(1)+size-2
        starting = size - 1
        for col in range(size - 1):
            consecutive = 0
            last = EMPTY
            for i in range(starting):
                point = self.board.board[start + i * (size)]
                if point != last:
                    consecutive = 1
                    last = point
                elif point == last:
                    consecutive += 1
                if consecutive == 5 and last == WHITE:

                    return "white"
                elif consecutive == 5 and last == BLACK:

                    return "black"
            start -= 1
            starting -= 1

        size = self.board.size
        starting = 1
        for row in range(size - 1, -1, -1):
            start = self.board.row_start(row + 1)
            consecutive = 0
            last = EMPTY
            for i in range(starting):
                point = self.board.board[start + i * (size+2)]
                if point != last:
                    consecutive = 1
                    last = point
                elif point == last:
                    consecutive += 1
                if consecutive == 5 and last == WHITE:

                    return "white"
                elif consecutive == 5 and last == BLACK:

                    return "black"
            starting += 1
        size = self.board.size
        start = self.board.row_start(1)+1
        starting = size-1

        for col in range(size-1):
            consecutive = 0
            last = EMPTY
            for i in range(starting):
                point = self.board.board[start + i * (size+2)]
                if point != last:
                    consecutive = 1
                    last = point
                elif point == last:
                    consecutive += 1
                if consecutive == 5 and last == WHITE:

                    return "white"
                elif consecutive == 5 and last == BLACK:

                    return "black"
            start += 1
            starting -= 1

        # vertical
        size = self.board.size
        start = self.board.row_start(self.board.size)

        for col in range(size):
            consecutive = 0
            last = EMPTY

            for i in range(size):
                point = self.board.board[start - i*(size+1)]
                if point != last:
                    consecutive = 1
                    last = point
                elif point == last:
                    consecutive += 1
                if consecutive == 5 and last == WHITE:

                    return "white"

                elif consecutive == 5 and last == BLACK:

                    return "black"
            start += 1

        # horizontal
        size = self.board.size
        for row in range(size-1, -1, -1):
            start = self.board.row_start(row + 1)
            consecutive = 0
            last = EMPTY
            for i in range(size):
                point = self.board.board[start + i]
                if point != last:
                    consecutive = 1
                    last = point
                elif point == last:
                    consecutive += 1
                if consecutive == 5 and last == WHITE:
                    return "white"
                elif consecutive == 5 and last == BLACK:
                    return "black"

        return "none"

    def gogui_check_neighbors_cmd(self, args: List[str]):

        # vertical
        # horizontal
        # diagonal_bt
        # diagonal_tb
        point_str = args[0]
        size = self.board.size
        # Convert point_str to a GO_POINT
        move_coord = move_to_coord(point_str, size)
        x, y = move_coord

        move_point = coord_to_point(move_coord[0], move_coord[1], size)
        player_color = self.board.get_color(move_point)

        opponent_color = ""
        captured_point = 0
        is_captured = False
        if player_color == 1:
            opponent_color = 2
        else:
            opponent_color = 1

        # left horizontal
        try:

            x = move_coord[0]
            y = move_coord[1]

            current_coord = (x-1, y)  # if manually change x and y change here
            current_point = coord_to_point(
                current_coord[0], current_coord[1], size)

            neighbor_color = self.board.get_color(
                current_point)  # gain color already

            if neighbor_color != 3 or neighbor_color != 0:  # BORDER = 3, empty = 0
                point = 0
                while x > 1:
                    x -= 1
                    # if manually change x and y change here
                    current_coord = (x, y)
                    current_point = coord_to_point(
                        current_coord[0], current_coord[1], size)
                    neighbor_color = self.board.get_color(
                        current_point)  # gain color already

                    if neighbor_color == opponent_color:
                        point += 1
                    # gain color already
                    elif neighbor_color == player_color:
                        self.respond("passed player color")
                        is_captured = True
                        break
                    else:
                        break

                if is_captured == True:
                    captured_point += point
            else:
                self.respond("no horizontal left neighbor")
        except Exception as e:
            # if error = reach left border
            self.respond("reach left border")

        # right horizontal
        try:

            x = move_coord[0]
            y = move_coord[1]
            current_coord = (x+1, y)  # if manually change x and y change here
            current_point = coord_to_point(
                current_coord[0], current_coord[1], size)

            neighbor_color = self.board.get_color(
                current_point)  # gain color already

            if neighbor_color != 3 or neighbor_color != 0:  # BORDER = 3, empty = 0
                point = 0
                while x < self.board.size:
                    x += 1
                    # if manually change x and y change here
                    current_coord = (x, y)
                    current_point = coord_to_point(
                        current_coord[0], current_coord[1], size)
                    neighbor_color = self.board.get_color(
                        current_point)  # gain color already

                    if neighbor_color == opponent_color:
                        point += 1
                    # gain color already
                    elif neighbor_color == player_color:
                        self.respond("passed player color")
                        is_captured = True
                        break
                    else:
                        break

                if is_captured == True:
                    captured_point += point
            else:
                self.respond("no horizontal right neighbor")
        except Exception as e:
            # if error = reach left border
            self.respond("reach right border")

            # bottom vertical
        try:
            x = move_coord[0]
            y = move_coord[1]
            current_coord = (x, y-1)  # if manually change x and y change here
            current_point = coord_to_point(
                current_coord[0], current_coord[1], size)

            neighbor_color = self.board.get_color(
                current_point)  # gain color already

            if neighbor_color != 3 or neighbor_color != 0:  # BORDER = 3, empty = 0
                point = 0
                while y < self.board.size:
                    y -= 1
                    # if manually change x and y change here
                    current_coord = (x, y)
                    current_point = coord_to_point(
                        current_coord[0], current_coord[1], size)
                    neighbor_color = self.board.get_color(
                        current_point)  # gain color already

                    if neighbor_color == opponent_color:
                        point += 1
                    # gain color already
                    elif neighbor_color == player_color:
                        self.respond("passed player color")
                        is_captured = True
                        break
                    else:
                        break

                if is_captured == True:
                    captured_point += point
            else:
                self.respond("no vertical right bottom neighbor")
        except Exception as e:
            # if error = reach bottom border
            self.respond("reach bottom border")

            # top vertical
        try:
            x = move_coord[0]
            y = move_coord[1]
            current_coord = (x, y+1)  # if manually change x and y change here
            current_point = coord_to_point(
                current_coord[0], current_coord[1], size)

            neighbor_color = self.board.get_color(
                current_point)  # gain color already

            if neighbor_color != 3 or neighbor_color != 0:  # BORDER = 3, empty = 0
                point = 0
                while y > 1:
                    y += 1
                    # if manually change x and y change here
                    current_coord = (x, y)
                    current_point = coord_to_point(
                        current_coord[0], current_coord[1], size)
                    neighbor_color = self.board.get_color(
                        current_point)  # gain color already

                    if neighbor_color == opponent_color:
                        point += 1
                    # gain color already
                    elif neighbor_color == player_color:
                        self.respond("passed player color")
                        is_captured = True
                        break
                    else:
                        break

                if is_captured == True:
                    captured_point += point
            else:
                self.respond("no verical top neighbor")
        except Exception as e:
            # if error = reach bottom border
            self.respond("reach bottom border")

            # diagonal top right
        try:
            x = move_coord[0]
            y = move_coord[1]
            # if manually change x and y change here
            current_coord = (x+1, y+1)
            current_point = coord_to_point(
                current_coord[0], current_coord[1], size)

            neighbor_color = self.board.get_color(
                current_point)  # gain color already

            if neighbor_color != 3 or neighbor_color != 0:  # BORDER = 3, empty = 0
                point = 0
                while x > 1 and y < self.board.size:
                    y += 1
                    x += 1
                    # if manually change x and y change here
                    current_coord = (x, y)
                    current_point = coord_to_point(
                        current_coord[0], current_coord[1], size)
                    neighbor_color = self.board.get_color(
                        current_point)  # gain color already

                    if neighbor_color == opponent_color:
                        point += 1
                    # gain color already
                    elif neighbor_color == player_color:
                        self.respond("passed player color")
                        is_captured = True
                        break
                    else:
                        break

                if is_captured == True:
                    captured_point += point
            else:
                self.respond("no diagonal top neighbor")
        except Exception as e:
            # if error = reach bottom border
            self.respond("reach top right border")

            # diagonal left bottom
        try:
            x = move_coord[0]
            y = move_coord[1]
            # if manually change x and y change here
            current_coord = (x-1, y-1)
            current_point = coord_to_point(
                current_coord[0], current_coord[1], size)

            neighbor_color = self.board.get_color(
                current_point)  # gain color already

            if neighbor_color != 3 or neighbor_color != 0:  # BORDER = 3, empty = 0
                point = 0
                while x > 1 and y < self.board.size:
                    y -= 1
                    x -= 1
                    # if manually change x and y change here
                    current_coord = (x, y)
                    current_point = coord_to_point(
                        current_coord[0], current_coord[1], size)
                    neighbor_color = self.board.get_color(
                        current_point)  # gain color already

                    if neighbor_color == opponent_color:
                        point += 1
                    # gain color already
                    elif neighbor_color == player_color:
                        self.respond("passed player color")
                        is_captured = True
                        break
                    else:
                        break

                if is_captured == True:
                    captured_point += point
            else:
                self.respond("no bottom left diagonal neighbor")
        except Exception as e:
            # if error = reach bottom border
            self.respond("reach bottom left border")

            # diagonal top left
        try:
            x = move_coord[0]
            y = move_coord[1]
            # if manually change x and y change here
            current_coord = (x-1, y+1)
            current_point = coord_to_point(
                current_coord[0], current_coord[1], size)

            neighbor_color = self.board.get_color(
                current_point)  # gain color already

            if neighbor_color != 3 or neighbor_color != 0:  # BORDER = 3, empty = 0
                point = 0
                while x > 1 and y < self.board.size:
                    y += 1
                    x -= 1
                    # if manually change x and y change here
                    current_coord = (x, y)
                    current_point = coord_to_point(
                        current_coord[0], current_coord[1], size)
                    neighbor_color = self.board.get_color(
                        current_point)  # gain color already

                    if neighbor_color == opponent_color:
                        point += 1
                    # gain color already
                    elif neighbor_color == player_color:
                        self.respond("passed player color")
                        is_captured = True
                        break
                    else:
                        break

                if is_captured == True:
                    captured_point += point
            else:
                self.respond("no bottom left diagonal neighbor")
        except Exception as e:
            # if error = reach bottom border
            self.respond("reach bottom left border")
            # diagonal top left
        try:
            x = move_coord[0]
            y = move_coord[1]
            # if manually change x and y change here
            current_coord = (x+1, y-1)
            current_point = coord_to_point(
                current_coord[0], current_coord[1], size)

            neighbor_color = self.board.get_color(
                current_point)  # gain color already

            if neighbor_color != 3 or neighbor_color != 0:  # BORDER = 3, empty = 0
                point = 0
                while x > 1 and y < self.board.size:
                    y -= 1
                    x += 1
                    # if manually change x and y change here
                    current_coord = (x, y)
                    current_point = coord_to_point(
                        current_coord[0], current_coord[1], size)
                    neighbor_color = self.board.get_color(
                        current_point)  # gain color already

                    if neighbor_color == opponent_color:
                        point += 1
                    # gain color already
                    elif neighbor_color == player_color:
                        self.respond("passed player color")
                        is_captured = True
                        break
                    else:
                        break

                if is_captured == True:
                    captured_point += point
            else:
                self.respond("no bottom left diagonal neighbor")
        except Exception as e:
            # if error = reach bottom border
            self.respond("reach bottom left border")

        self.respond("captured_point: " + str(captured_point))
        return captured_point

    def gogui_rules_legal_moves_cmd(self, optional_param: str = None) -> None:
        """ Implement this function for Assignment 1 """
        self.optional_param = optional_param
        try:
            # Checking part
            # Check for color
            board_color = "b"
            gtp_moves = []
            # Check if game is over
            game_result = self.gogui_rules_final_result_cmd([])
            if game_result != "unknown":
                return []

            color = color_to_int(board_color)
            legal_moves = GoBoardUtil.generate_legal_moves(self.board, color)

            # Format the moves
            for move in legal_moves:
                coords = point_to_coord(move, self.board.size)
                gtp_moves.append(format_point(coords))

            # Sort the moves for consistency
            sorted_moves = " ".join(sorted(gtp_moves))

            self.respond(sorted_moves)
        except Exception as e:
            self.respond("Error: {}".format(str(e)))

    # basically a copy of the above function but return
    def gogui_rules_legal_moves_cmd_return(self, optional_param: str = None) -> None:
        """ Implement this function for Assignment 1 """
        self.optional_param = optional_param
        try:
            # Checking part
            # Check for color
            board_color = "b"
            if board_color not in ('b', 'w'):
                self.error("Invalid color. Use 'b' or 'w'.")
                return
            gtp_moves = []
            # Check if game is over
            game_result = self.gogui_rules_final_result_cmd([])
            if game_result != "unknown":
                return gtp_moves

            color = color_to_int(board_color)
            legal_moves = GoBoardUtil.generate_legal_moves(self.board, color)

            # Format the moves
            for move in legal_moves:
                coords = point_to_coord(move, self.board.size)
                gtp_moves.append(format_point(coords))

            return gtp_moves
        except Exception as e:
            self.respond("Error: {}".format(str(e)))

    def play_cmd(self, args: List[str]) -> None:
        """
        Modify this function for Assignment 1.
        Play a move args[1] for the given color args[0] in {'b', 'w'}.
        """
        try:
            board_color = args[0].lower()
            board_move = args[1]
            color = color_to_int(board_color)

            # Get the list of legal moves for the current player

            legal_moves = self.gogui_rules_legal_moves_cmd_return("something")
            # self.respond(str(legal_moves))
            # self.respond("debug1")

            if args[1].lower() == "pass":
                self.board.play_move(PASS, color)
                self.board.current_player = opponent(color)
                self.respond()
                return
            # self.respond("debug2")
            # color check
            if self.board.current_player != color:
                self.respond("Illegal Move: {}".format(board_color))
            #     return
            # self.respond("debug3")
            # occupication and also coordiante check
            if board_move not in legal_moves:
                self.respond("Illegal Move: {}".format(board_move))
                return
            # self.respond("debug4")
            coord = move_to_coord(args[1], self.board.size)
            move = coord_to_point(coord[0], coord[1], self.board.size)

            if not self.board.play_move(move, color):
                self.respond("Illegal Move: {}".format(board_move))
                return
            else:
                self.debug_msg(
                    "Move: {}\nBoard:\n{}\n".format(board_move, self.board2d())
                )
            self.respond()
        except Exception as e:
            self.respond("Error: {}".format(str(e)))

    def genmove_cmd(self, args: List[str]) -> None:
        """ 
        Modify this function for Assignment 1.
        Generate a move for color args[0] in {'b','w'}.
        """
        board_color = args[0].lower()
        color = color_to_int(board_color)
        move = self.go_engine.get_move(self.board, color)
        move_coord = point_to_coord(move, self.board.size)
        move_as_string = format_point(move_coord)

        if self.gogui_rules_final_result_cmd([]) == "black":
            self.respond("resign")
        elif self.gogui_rules_final_result_cmd([]) == "white":
            self.respond("resign")
        elif self.gogui_rules_final_result_cmd([]) == "draw":
            self.respond("pass")
        elif self.board.is_legal(move, color):  # change
            self.board.play_move(move, color)
            self.respond(move_as_string)
        else:
            self.respond("Illegal move: {}".format(move_as_string))

    def gogui_captured_check_cmd(self, args: List[str]) -> None:
        self.respond()

    def gogui_rules_captured_count_cmd(self, args: List[str]) -> None:
        """ 
        Modify this function for Assignment 1.
        Respond with the score for white, an space, and the score for black.
        """
        self.respond("0 0")

    """
    ==========================================================================
    Assignment 1 - game-specific commands end here
    ==========================================================================
    """


def point_to_coord(point: GO_POINT, boardsize: int) -> Tuple[int, int]:
    """
    Transform point given as board array index 
    to (row, col) coordinate representation.
    Special case: PASS is transformed to (PASS,PASS)
    """
    if point == PASS:
        return (PASS, PASS)
    else:
        NS = boardsize + 1
        return divmod(point, NS)


def format_point(move: Tuple[int, int]) -> str:
    """
    Return move coordinates as a string such as 'A1', or 'PASS'.
    """
    assert MAXSIZE <= 25
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if move[0] == PASS:
        return "PASS"
    row, col = move
    if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
        raise ValueError
    return column_letters[col - 1] + str(row)


def move_to_coord(point_str: str, board_size: int) -> Tuple[int, int]:
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.
    Raises ValueError if point_str is invalid
    """
    if not 2 <= board_size <= MAXSIZE:
        raise ValueError("board_size out of range")
    s = point_str.lower()
    if s == "pass":
        return (PASS, PASS)
    try:
        col_c = s[0]
        if (not "a" <= col_c <= "z") or col_c == "i":
            raise ValueError
        col = ord(col_c) - ord("a")
        if col_c < "i":
            col += 1
        row = int(s[1:])
        if row < 1:
            raise ValueError
    except (IndexError, ValueError):
        raise ValueError("invalid point: '{}'".format(s))
    if not (col <= board_size and row <= board_size):
        raise ValueError("point off board: '{}'".format(s))
    return row, col


def color_to_int(c: str) -> int:
    """convert character to the appropriate integer code"""
    color_to_int = {"b": BLACK, "w": WHITE, "e": EMPTY, "BORDER": BORDER}
    return color_to_int[c]
