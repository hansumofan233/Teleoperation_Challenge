import sys
if sys.prefix == '/home/cnbot/miniconda3/envs/env_6dofarms':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/cnbot/Teleoperation_Challenge/install/robotiq_85_driver'
