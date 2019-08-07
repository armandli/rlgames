import subprocess
import re
import h5py

from rlgames.common_types import Player
from rlgames.game_base import Move
from rlgames.goboard import GameState
from rlgames.goscore import AreaScore
from rlgames.agents.predict import load_prediction_agent
from rlgames.agents.termination import PassWhenOpponentPasses, TerminationAgent
from rlgames.gtp.board import gtp_position_to_coord, coord_to_gtp_position
from rlgames.gtp.utils import SGFWriter

from rlgames.util import print_board

class LocalGtpBot:
  def __init__(self, gobot, termination=None, handicap=0, opponent='gnugo', output_sgf='out.sgf', our_color='b'):
    self.bot = gobot
    self.handicap = handicap
    self.stopped = False
    self.game_state = GameState.new_game(19)
    self.sgf = SGFWriter(output_sgf)
    self.our_color = Player.black if our_color == 'b' else Player.white
    self.their_color = self.our_color.other
    cmd = self.opponent_cmd(opponent)
    pipe = subprocess.PIPE
    self.gtp_stream = subprocess.Popen(cmd, stdin=pipe, stdout=pipe) #allow read/write to gtp stream through command line

  @staticmethod
  def opponent_cmd(opponent):
    if opponent == 'gnugo':
      return ['gnugo', '--mode', 'gtp']
    elif opponent == 'pachi':
      return ['pachi']
    else:
      raise ValueError('Unknown bot name {}'.format(opponent))

  def send_command(self, cmd):
    self.gtp_stream.stdin.write(cmd.encode('utf-8'))
    self.gtp_stream.stdin.flush()

  def get_response(self):
    success = False
    result = ''
    while not success:
      line = self.gtp_stream.stdout.readline().decode('utf-8')
      if line[0] == '=':
        success = True
        line = line.strip()
        result = re.sub('^= ?', '', line)
    return result

  def command_and_response(self, cmd):
    self.send_command(cmd)
    return self.get_response()

  def run(self):
    self.command_and_response('boardsize 19\n')
    self.set_handicap()
    self.play()
    self.sgf.write_sgf()

  def set_handicap(self):
    if self.handicap == 0:
      self.command_and_response('komi 7.5\n')
      self.sgf.append('KM[7.5]\n')
    else:
      stones = self.command_and_response('fixed_handicap {}\n'.format(self.handicap))
      sgf_handicap = 'HA[{}]AB'.format(self.handicap)
      for pos in stones.split(' '):
        move = gtp_position_to_coord(pos)
        self.game_state = self.game_state.apply_move(move)
        sgf_handicap = sgf_handicap + '[' + self.sgf.coordinates(move) + ']'
        self.sgf.append(sgf_handicap + '\n')

  def play(self):
    while not self.stopped:
      if self.game_state.nplayer == self.our_color:
        self.play_our_move()
      else:
        self.play_their_move()
      print_board(self.game_state.board)
      print('Estimated result: ')
      score = AreaScore(self.game_state.board)
      print(score)

  def play_our_move(self):
    move = self.bot.select_move(self.game_state)
    self.game_state = self.game_state.apply_move(move)
    our_name = self.our_color.name
    our_letter = our_name[0].upper()
    sgf_move = ''
    if move.is_pass:
      self.command_and_response('play {} pass\n'.format(our_name))
    elif move.is_resign:
      self.command_and_response('play {} resign\n'.format(our_name))
    else:
      pos = coord_to_gtp_position(move)
      self.command_and_response('play {} {}\n'.format(our_name, pos))
      sgf_move = self.sgf.coordinates(move)
    self.sgf.append(';{}[{}]\n'.format(our_letter, sgf_move))

  def play_their_move(self):
    their_name = self.their_color.name
    their_letter = their_name[0].upper()
    pos = self.command_and_response('genmove {}\n'.format(their_name))
    if pos.lower() == 'resign':
      self.game_state = self.game_state.apply_move(Move.resign())
      self.stopped = True
    elif pos.lower() == 'pass':
      self.game_state = self.game_state.apply_move(Move.pass_turn())
      self.sgf.append(';{}[]\n'.format(their_letter))
      if self.game_state.pmove.is_pass:
        self.stopped = True
    else:
      move = gtp_position_to_coord(pos)
      self.game_state = self.game_state.apply_move(move)
      self.sgf.append(';{}[{}]\n'.format(their_letter, self.sgf.coordinates(move)))

if __name__ == '__main__':
  agent_file = '/home/armandli/rlgames/data/agents/sl_medium_sevenplane_9.h5'
  bot = load_prediction_agent(h5py.File(agent_file, 'r'))
  gtp_bot = LocalGtpBot(gobot = bot, termination=PassWhenOpponentPasses(), handicap=0, opponent='gnugo')
  gtp_bot.run()
