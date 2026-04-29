#Script to switch between conda environments
#This is due to conflicts between deimos and ipaPy2 dependancies
#Date created: 13/03/2025
#Script author: Jessica M. O'Loughlin (s1907024@ed.ac.uk)
#Supervisor: Prof. Karl Burgess (karl.burgess@ed.ac.uk)

import subprocess
from datetime import datetime #Get current date and time

print("Master Script initiated")
startTime = datetime.now()
print("===============================")
print("Master Script startTime:", startTime)
print("===============================")

print("DEIMoS-based  script initiated")

# subprocess.run('conda run -n deimos python deimos_simple.py', shell = True)
subprocess.run('conda run -n deimos python 20250313_DEIMoSProcessingSteps.py', shell = True)

print("DEIMoS-based script complete")
# =============================================================================
# print("ipaPy2-based script initiated")
# 
# # subprocess.run('conda run -n ipaPy2_env python ipaPy2_simple.py', shell = True)
# subprocess.run('conda run -n ipaPy2_env python 20250313_ipaPy2ProcessingSteps.py', shell = True)
# 
# print("ipaPy2-based script complete")
# 
# print("Master Script complete")
# =============================================================================

print("===============================")
print("Master Script stopTime:", datetime.now())
print("Total process run time:", datetime.now() - startTime)
print("===============================")
