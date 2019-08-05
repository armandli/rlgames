import subprocess

cmd=['gnugo', '--mode', 'gtp']
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
proc.stdin.write('boardsize 9\n'.encode('utf-8'))
proc.stdin.flush()
msg = proc.stdout.readline().decode('utf-8').strip()
print(msg)

proc.stdin.close()
proc.terminate()
