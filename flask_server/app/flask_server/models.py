from flask_server.shared import db, jwt, ma


class Site(db.Model):  
    __tablename__ = 'nsb_site'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

    def __init__(self, name):
        self.name = name
        self.email = email

    def save(self):
        db.session.add(self)
        db.session.commit()

    def __repr__(self):
        return self.name


class Device(db.Model):  
    __tablename__ = 'nsb_device'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    status = db.Column(db.Boolean, default=True, nullable=False)
    lat = db.Column(db.Float, nullable=False)
    lon = db.Column(db.Float, nullable=False)

    def __init__(self, name, status, lat, lon):
        self.name = name
        self.status = status
        self.lat = lat
        self.lon = lon

    def save(self):
        db.session.add(self)
        db.session.commit()

    def __repr__(self):
        return self.name
