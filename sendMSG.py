import smtplib
import imghdr
from email.message import EmailMessage
from twilio.rest import Client

def writeMsg(device_name, nomask_count, mask_count, close_objects, datetime_ist):
    msg="**Social Distancing and Face Mask System Alert** \n\n"
    msg+=f"Camera ID: {device_name}" + "\n\n"
    msg+="Status: Danger! \n\n"
    msg+="No_Mask Count: "+str(nomask_count)+" \n"
    msg+="Mask Count: "+str(mask_count)+" \n"
    msg+=f"Social Distancing Violations: {len(close_objects)}"+" \n"
    msg+="Date-Time of alert: \n"+datetime_ist.strftime('%Y-%m-%d %H:%M:%S %Z')
    return msg

def sendSMS(msg):
    with open('account.txt', 'r') as f:
        account = f.read().split(',')
        SID = account[2]
        auth_token = account[3]

    client = Client(SID, auth_token)
    res = client.messages.create(body=msg, from_=account[5], to=account[4])
    print(res.sid)

def sendEmail(msg, image, Reciever_Email="nguyendodangquang@gmail.com"):
    with open('account.txt', 'r') as f:
        account = f.read().split(',')
        Sender_Email = account[0]
        Password = account[1]

    newMessage = EmailMessage()                         
    newMessage['Subject'] = "Social Distancing and Facemask Notifier" 
    newMessage['From'] = Sender_Email                   
    newMessage['To'] = Reciever_Email                   
    newMessage.set_content(msg) 

    with open(image, 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Sender_Email, Password)   
        smtp.send_message(newMessage)

    print(print("mail sent to:",Reciever_Email))