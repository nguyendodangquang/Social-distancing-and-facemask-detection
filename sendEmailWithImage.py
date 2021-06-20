import smtplib
import imghdr
from email.message import EmailMessage

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