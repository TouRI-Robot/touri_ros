import stretch_body.arm

a = stretch_body.arm.Arm()
a.motor.disable_sync_mode()
if not a.startup():
    exit() # failed to start arm!

a.move_to("stretch_gripper", 100)
a.push_command()

a.stop()