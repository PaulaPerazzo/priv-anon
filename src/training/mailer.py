### mailer.py gmail

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(subject):
    msg = MIMEMultipart()

    email = "mpps@cin.ufpe.br"
    password = "ptfe ptnq xupt vqbg"
    message = f"your model for {subject} is ready"

    msg['From'] = email
    msg['To'] = email
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(email, password)
    text = msg.as_string()
    server.sendmail(email, email, text)
    server.quit()

    print('Email sent')
