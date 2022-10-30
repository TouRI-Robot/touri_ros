import stretch_body.arm
import time

a = stretch_body.arm.Arm()
a.motor.disable_sync_mode()
a.startup(threaded=False)

print(a.motor.transport.failout_counter)

a.stop()

