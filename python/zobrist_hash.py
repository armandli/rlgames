import random
from gotypes import Player, Point

#Create perfect hash value for each go board state

def to_python(player_state):
  if player_state is None:
    return 'None'
  if player_state == Player.black:
    return Player.black
  else:
    return Player.white

MAX63 = 0x7fffffffffffffff

table = dict()
empty_board = 0
for row in range(1, 21):
  for col in range(1, 21):
    for state in (Player.black, Player.white):
      code = random.randint(0, MAX63)
      table[Point(row, col), state] = code

print('from gotypes import Player, Point')
print('')
print("__all__ = ['HASH_CODE', 'EMPTY_BOARD']")
print('')
print('HASH_CODE = {')
for (pt, state), hash_code in table.items():
  print('  (%r, %s): %r,' % (pt, to_python(state), hash_code))
print('}')
print('')
print('EMPTY_BOARD = %d' % (empty_board,))
