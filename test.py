#! /usr/bin/python
import sys
import time

def test(i):
	try:
		for i in range(100):
			sys.stdout.write("\r  %d  %d"%i %i)
			time.sleep(0.1)
			
			sys.stdout.flush()
		a=6/i

	except:
		 test(i+1)
	else:
		print "c"


