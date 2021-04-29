from flask import current_app, url_for, render_template
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer


mail = Mail()


def generate_confirmation_token(email):
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    return serializer.dumps(email, salt=current_app.config['SECURITY_PASSWORD_SALT'])


def confirm_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    try:
        email = serializer.loads(
            token,
            salt=current_app.config['SECURITY_PASSWORD_SALT'],
            max_age=expiration
        )
    except:
        return False
    return email


def send_email(to, subject, template):
    msg = Message(
        subject,
        recipients=[to],
        html=template,
        sender=current_app.config['MAIL_DEFAULT_SENDER']
    )
    mail.send(msg)


def send_confirmation_email(to, token):
    confirm_url = url_for(
        'bp_user.confirm_email',
        token=token,
        _external=True)
    html = render_template(
        'email/activate.html',
        confirm_url=confirm_url)
    send_email(to, "NSB account - Please confirm your email.", html)
