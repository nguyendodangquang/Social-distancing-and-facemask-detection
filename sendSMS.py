from twilio.rest import Client

def sendSMS(msg):
    with open('account.txt', 'r') as f:
        account = f.read().split(',')
        SID = account[2]
        auth_token = account[3]

    client = Client(SID, auth_token)
    res = client.messages.create(body=msg, from_=account[5], to=account[4])
    print(res.sid)