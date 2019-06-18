from __future__ import print_function

import test
#UNIFORM,ERRORB,COMVAR,EPOCHS,FILENAME
for i in range(10):
	K=test.test_code(True,i*6,False,10,"otpt2.txt")
	K.run()

for i in range(10):
	J=test.test_code(False,i*6,True,10,"otpt2.txt")
	J.run()

for i in range(10):
	L=test.test_code(False,i*6,False,10,"otpt2.txt")
	L.run()

