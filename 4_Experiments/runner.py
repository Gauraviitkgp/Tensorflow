from __future__ import print_function

import test
#UNIFORM,ERRORB,COMVAR,EPOCHS,FILENAME
for i in range(20):
	K=test.test_code(True,i*3,False,6,"otpt.txt")
	K.run()

for i in range(20):
	J=test.test_code(False,i*3,True,6,"otpt.txt")
	J.run()

for i in range(20):
	L=test.test_code(False,i*3,False,6,"otpt.txt")
	L.run()

