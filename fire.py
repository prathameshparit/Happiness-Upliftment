import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime

class Uploader:
    def __init__(self):
        # Fetching the service account key JSON file
        cred = credentials.Certificate("serviceKey.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                "databaseURL": "https://multimodalai-default-rtdb.firebaseio.com/"
            })

        # Saving the data
        ref = db.reference("contact/")
        self.users_ref = ref.child("data")
    
    def upload(self, label, ans1, ans2, ans3):
        now = str(datetime.now()).split(".")[0].replace(" ", "--").replace(":", "-")

        self.users_ref.update({
            now: {
                "label": label,
                "Q1": ans1,
                "Q2:": ans2,
                "Q3": ans3,
            }
        })

upl = Uploader()
upl.upload('financial', "yes", "no", 'yes')